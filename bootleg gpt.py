import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tokenizer import BPETokenizer


@dataclass
class BootlegGPTConfig():
    vocab_size: int = 50257  # Size of vocabulary
    n_ctx: int = 1024  # Max context length
    n_embd: int = 768  # Embedding size
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_inner: int = None  # Dimensionality of feedforward layers, default is 4 * n_embd
    drop: int = 0.1  # Dropout probability
    layer_norm_eps: int = 1e-5  # Epsilon for layer norm layers
    bias: bool = True  # Include bias in linear layers
    device: str = None  # Which device to run on

    def __post_init__(self):
        if self.n_inner == None:
            self.n_inner = 4 * self.n_embd
        if self.device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd is not a multiple of n_head")


class LayerNorm(nn.Module):
    '''Normalizes across the channel dimension'''

    def __init__(self, config):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.gamma = torch.ones(config.n_embd, device=config.device)
        self.beta = torch.zeros(config.n_embd, device=config.device)

    def forward(self, x):
        mean = x.mean(2, keepdim=True)
        std = x.var(2, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        return x * self.gamma + self.beta


class ScaledDotProductAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Query, key, value transforms
        self.query = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=config.bias)

        self.bitmask = torch.tril(torch.ones(self.config.n_ctx, self.config.n_ctx, device=self.config.device))

        # Attention dropout
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        # Attention score -- Dot product of query with all keys and divide by sqrt(d_k)
        alpha = q @ k.transpose(1, 2) * k.shape[-1]**-0.5  # (b, t, c_head) @ (b, c_head, t) --> (b, t, t)
        # Attention masking
        alpha = alpha.masked_fill(self.bitmask == 0, float("-inf"))
        alpha = F.softmax(alpha, dim=-1)
        # Randomly dropout from the softmax
        alpha = self.dropout(alpha)
        v = self.value(x)
        h = alpha @ v  # (b, t, t) @ (b, t, c) --> (b, t, c)
        return h


class MultiHeadAttention(nn.Module):
    '''Concatenation of parallel scaled dot product attention layers'''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([ScaledDotProductAttention(config) for _ in range(config.n_head)])
        self.linear = nn.Linear(config.n_embd, config.n_embd)
        # Residual dropout
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        # Concatenate in the channel dimension (final c = n_embd)
        out = torch.cat([head(x) for head in self.heads], dim=2)
        out = self.linear(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    '''Two linear transformations with a ReLU activation in between'''

    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.n_embd, config.n_inner)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return self.dropout(x)

class Block(nn.Module):
    '''Single transformer block'''
    
    def __init__(self, config):
        super().__init__()
        # Uses pre-normalization
        self.ln_1 = LayerNorm(config)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config)
        self.ffw = FeedForward(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.ln_2(x)
        return self.ffw(x)


# Based on the nanogpt implementation of gpt2
class BootlegGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Word token embedding / word position embedding lookup tables
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)

        # Dropout layer
        self.drop = nn.Dropout(config.drop)

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm + linear layer
        self.layernorm = LayerNorm(config)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)
    
    def forward(self, idx, targets=None):
        # idxs come in with a batch and time dimension: b, t
        b, t = idx.size()

        # After passing idx to the embedding tables, channel dimension gets added (n_embd)
        tok_emb = self.wte(idx)  # (b, t) --> (b, t, n_embd)
        pos = torch.arange(t, device=idx.device)
        pos_emb = self.wpe(pos)  # (t) --> (t, n_embd)

        # Embedding dropout
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        # LM head
        logits = self.linear(x)  # (b, t, n_embd) --> (b, t, vocab_size)

        if targets is not None:
            # Training
            logits = logits.view(-1, self.config.vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            # Inference
            loss = None

        return logits, loss


if __name__ == "__main__":
    # Tests
    c = BootlegGPTConfig(vocab_size=500, n_ctx=4, n_embd=8, n_head=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    batch_sz = 6
    with open("wikitext-103-raw/wiki.valid.raw", encoding="utf-8") as f:
        data = f.read()
    tokenizer = BPETokenizer("vocab.json", "merges.txt")
    data = tokenizer(data)
    split = int(0.9*len(data))
    train_data = data[:split]
    val_data = data[split:]
    def get_batch():
        # Sample random indices
        idxs = torch.randint(len(train_data) - c.n_ctx, (batch_sz,))
        x = torch.stack([train_data[idx:idx+c.n_ctx] for idx in idxs]).to(device)
        y = torch.stack([train_data[idx+1:idx+c.n_ctx+1] for idx in idxs]).to(device)
        return x, y
    # b x t x c -- 6 x 4 x 8
    model = BootlegGPT(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = 0
    for i in range(1000):
        if i % 100 == 0:
            print(f"Step {i}: Loss={loss}")
        xb, yb = get_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
