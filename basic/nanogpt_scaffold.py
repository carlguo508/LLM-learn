"""
nanoGPT-style scaffold: data loading + training skeleton.
Architecture (model.py concerns) is left empty for you to fill in.

Run this file once to verify data loading works before touching the model.
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Config
# =====================================================================

class Config:
    # data
    block_size = 256        # context length
    batch_size = 64

    # model (Checkpoint 1.6 target: ~10M params)
    n_layer = 6
    n_head = 6
    d_model = 384
    dropout = 0.0           # keep 0 until everything works, then try 0.1
    # vocab_size is set after loading data

    # training
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 3e-4
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # repro
    seed = 1337


cfg = Config()
torch.manual_seed(cfg.seed)
print(f"Using device: {cfg.device}")


# =====================================================================
# Data: download TinyShakespeare and build a char-level encoder
# =====================================================================

DATA_PATH = 'input.txt'
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists(DATA_PATH):
    print(f"Downloading TinyShakespeare to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Build the char-level vocab. This is NOT a real tokenizer, just a dict.
chars = sorted(list(set(text)))
cfg.vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

# Encode the entire dataset, split 90/10
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset: {len(text)} chars, vocab_size={cfg.vocab_size}")
print(f"Train: {len(train_data)} tokens, Val: {len(val_data)} tokens")


def get_batch(split):
    """Return a random batch (x, y) where y is x shifted by 1."""
    src = train_data if split == 'train' else val_data
    # Pick batch_size random starting indices
    ix = torch.randint(len(src) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# =====================================================================
# Model: YOUR JOB STARTS HERE
# =====================================================================
#
# Fill in this class following the checkpoint list.
#
# Checkpoint 1.2: token embedding + position embedding
# Checkpoint 1.3: single-head self-attention (write it standalone first)
# Checkpoint 1.4: multi-head self-attention
# Checkpoint 1.5: FFN + Block (with pre-norm + residual)
# Checkpoint 1.6: stack N blocks, final LayerNorm, LM head
#

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        # nn.ModuleList，里面装 n_head 个 Head(head_size)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # 让每个 head 都对 x 做 forward，结果 cat 起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 过 proj
        out = self.proj(out)
        # 过 dropout
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.W_q = nn.Linear(cfg.d_model, head_size)
        self.W_k = nn.Linear(cfg.d_model, head_size)
        self.W_v = nn.Linear(cfg.d_model, head_size)
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, x):  # x: (B, T, d_model)
        B, T, _ = x.shape
        Q = self.W_q(x) # Q: (B, T, head_size)
        K = self.W_k(x) # K: (B, T, head_size)
        V = self.W_v(x) # V: (B, T, head_size)
        weight = Q @ K.transpose(-2, -1) * self.head_size **-0.5 # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ V
        return out
        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.sa  = MultiHeadAttention(cfg.n_head, cfg.d_model // cfg.n_head)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # 注意 + ：残差连接
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.d_model)
        # Checkpoint 1.6: define the stack of blocks, final ln, lm_head
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        

    # forward: for training
    # forward(idx, targets=None) should:
    #   - take idx of shape (B, T)
    #   - return (logits, loss)
    #     logits: (B, T, vocab_size)
    #     loss:   scalar if targets is given, else None
    def forward(self, idx, targets=None):
        # shape flow (B, T) -> (B, T, d_model) -> ... -> (B, T, vocab_size)
        B, T = idx.shape
        # (B, T, d_model)
        tok_emb = self.token_embedding_table(idx)
        # [0, T - 1]
        position_idx = torch.arange(T, device=idx.device)
        # (T, d_model)
        pos_emb = self.position_embedding_table(position_idx)
        # (B, T, d_model)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)  # final LayerNorm
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        
        return logits, loss

    # generate: for inference
    # generate(idx, max_new_tokens) should:
    #   - take idx of shape (B, T_start)
    #   - autoregressively append max_new_tokens tokens
    #   - return idx of shape (B, T_start + max_new_tokens)
    #   - (Stage 1 version: re-run the full context every step. Naive on purpose.)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T_start)
        for _ in range(max_new_tokens):
            # 1. 截断到最多 block_size
            idx_cond = idx[:, -self.cfg.block_size:]
            
            # 2. forward，丢掉 loss
            logits, _ = self(idx_cond)
            
            # 3. 只取最后一个位置
            logits = logits[:, -1, :]            # (B, vocab_size)
            
            # 4. 应用 temperature
            logits = logits / temperature
            
            # 5. (可选) top_k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 6. softmax → 概率
            probs = F.softmax(logits, dim=-1)
            
            # 7. 采样一个 token
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            
            # 8. 拼到 idx 后面
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx



# =====================================================================
# Training loop (works once GPT is implemented)
# =====================================================================

@torch.no_grad()
def estimate_loss(model):
    """Average loss over eval_iters batches for both splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    model = GPT(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        x, y = get_batch('train')
        _, loss = model(x, y)
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


# =====================================================================
# Sanity check: make sure data loading works before you touch the model
# =====================================================================

if __name__ == '__main__':
    # Verify get_batch produces correctly shifted pairs
    x, y = get_batch('train')
    print(f"\nbatch x shape: {x.shape}")  # expect (64, 256)
    print(f"batch y shape: {y.shape}")
    print(f"x[0, :20]: {x[0, :20].tolist()}")
    print(f"y[0, :20]: {y[0, :20].tolist()}")
    print(f"y should equal x shifted left by 1 -> y[0,0] should equal x[0,1]:")
    print(f"  x[0,1] = {x[0,1].item()}, y[0,0] = {y[0,0].item()}")
    assert x[0, 1].item() == y[0, 0].item(), "shift is wrong"

    print(f"\nsample decoded x[0, :100]:")
    print(repr(decode(x[0, :100].tolist())))

    # Once you've implemented GPT, uncomment:
    # model = train()
    # context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    # print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

