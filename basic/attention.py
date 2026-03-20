import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        Q = self.W_q(x)  # [batch_size, seq_len, embed_dim]
        K = self.W_k(x)  # [batch_size, seq_len, embed_dim]
        V = self.W_v(x)  # [batch_size, seq_len, embed_dim]

        K = K.transpose(-2, -1)                        # [batch, embed_dim, seq_len]
        attn_scores = Q @ K / self.scale               # [batch, seq_len, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]
        out = attn_weights @ V                         # [batch, seq_len, embed_dim]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape

        Q = self.W_q(x)  # [batch_size, seq_len, embed_dim]
        K = self.W_k(x)  # [batch_size, seq_len, embed_dim]
        V = self.W_v(x)  # [batch_size, seq_len, embed_dim]

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)  # [batch, num_heads, seq_len, head_dim]
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)  # [batch, num_heads, seq_len, head_dim]
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)  # [batch, num_heads, seq_len, head_dim]

        attn_scores = Q @ K.transpose(-2, -1) / self.scale   # [batch, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)        # [batch, num_heads, seq_len, seq_len]
        out = attn_weights @ V                               # [batch, num_heads, seq_len, head_dim]

        out = out.transpose(-3, -2).reshape(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim]
        return self.W_o(out)


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    x = torch.randn(2, 10, embed_dim)  # batch=2, seq_len=10, embed_dim=512

    sa = SelfAttention(embed_dim)
    print(sa(x).shape)  # expected: torch.Size([2, 10, 512])

    mha = MultiHeadAttention(embed_dim, num_heads)
    print(mha(x).shape)  # expected: torch.Size([2, 10, 512])