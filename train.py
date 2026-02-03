import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------
# Hyperparameters (same spirit as Karpathy)
# -----------------------------
batch_size = 64      # sequences processed in parallel
block_size = 256       # context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 384         # embedding dimension
n_head = 6          # number of attention heads
n_layer = 6  
dropout=0.2       # number of transformer blocks


torch.manual_seed(1337)

# -----------------------------
# Load your Friends-SWE dataset
# -----------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset length (chars):", len(text))

# -----------------------------
# Character-level tokenizer
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------
# Train / validation split
# -----------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -----------------------------
# Batch sampler
# -----------------------------
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# -----------------------------
# Loss estimation (IMPORTANT)
# -----------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#
# Head - Single Head Self_Attention
#
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # causal mask (lower triangular)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)

        # causal masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # normalize
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)

        return out

#
# MultiHeadAttention
#
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size)
        out = self.proj(out)                                 # (B, T, n_embed)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 *n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# -----------------------------
# Bigram Language Model
# -----------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Direct lookup: token -> logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])  
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.shape

        token_embeds = self.token_embedding_table(idx)  # (B, T, C)
        pos_embeds = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embeds + pos_embeds  # (B, T, C) 
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x)             # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]      # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------
# Initialize model + optimizer
# -----------------------------
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------
# Training loop
# -----------------------------
for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------
# Sample from the model
# -----------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
