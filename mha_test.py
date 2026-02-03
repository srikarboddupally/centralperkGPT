import torch
import torch.nn as nn
from torch.nn import functional as F

# replicate relevant hyperparams
batch_size = 4
block_size = 8
n_embed = 32
dropout = 0.2
n_head = 6

torch.manual_seed(1337)

def debug(msg, obj=None):
    print(msg)
    if obj is not None:
        print(obj)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

if __name__ == '__main__':
    head_size = n_embed // n_head
    m = MultiHeadAttention(n_head, head_size)
    print(f"MultiHeadAttention: num_heads={n_head}, head_size={head_size}, proj_in={head_size*n_head}, proj_out={n_embed}")

    x = torch.randn(batch_size, block_size, n_embed)
    out = m(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    # simple correctness checks
    assert out.shape == (batch_size, block_size, n_embed)
    print("Forward pass successful ✅")
