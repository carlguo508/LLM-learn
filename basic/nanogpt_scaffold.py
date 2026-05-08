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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # repro
    seed = 1337


cfg = Config()
torch.manual_seed(cfg.seed)


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

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.d_model)
        # TODO Checkpoint 1.6: define the stack of blocks, final ln, lm_head

    # forward: for training
    # forward(idx, targets=None) should:
    #   - take idx of shape (B, T)
    #   - return (logits, loss)
    #     logits: (B, T, vocab_size)
    #     loss:   scalar if targets is given, else None
    def forward(self, idx, targets=None):
        # TODO: shape flow (B, T) -> (B, T, d_model) -> ... -> (B, T, vocab_size)
        B, T = idx.shape
        # (B, T, d_model)
        tok_emb = self.token_embedding_table(idx)
        # [0, T - 1]
        position_idx = torch.arange(T, device=idx.device)
        # (T, d_model)
        pos_emb = self.position_embedding_table(position_idx)
        # (B, T, d_model)
        x = tok_emb + pos_emb
        return x, None

    # generate: for inference
    # generate(idx, max_new_tokens) should:
    #   - take idx of shape (B, T_start)
    #   - autoregressively append max_new_tokens tokens
    #   - return idx of shape (B, T_start + max_new_tokens)
    #   - (Stage 1 version: re-run the full context every step. Naive on purpose.)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # TODO Checkpoint 1.8
        raise NotImplementedError


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
    model = GPT(cfg).to(cfg.device)
    x, y = get_batch('train')
    out, _ = model(x)
    print(out.shape)   # 期望: torch.Size([64, 256, 384])

