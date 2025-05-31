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
        self.beta = torch.zeros(config.n_embd, device=config.device) if config.bias else None

    def forward(self, x):
        mean = x.mean(2, keepdim=True)
        std = x.var(2, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        if self.beta != None:
            return x * self.gamma + self.beta
        else:
            return x * self.gamma


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
        T = x.shape[1]  # Extract sequence length
        q = self.query(x)
        k = self.key(x)
        # Attention score -- Dot product of query with all keys and divide by sqrt of key dim
        alpha = q @ k.transpose(1, 2) * k.shape[-1]**-0.5  # (b, t, c_head) @ (b, c_head, t) --> (b, t, t)
        # Attention masking
        alpha = alpha.masked_fill(self.bitmask[:T, :T] == 0, float("-inf"))
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
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
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
        self.lin1 = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return self.dropout(x)

class Block(nn.Module):
    '''Single transformer block'''
    
    def __init__(self, config):
        super().__init__()
        # Uses pre-normalization: norm and add
        self.ln_1 = LayerNorm(config)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config)
        self.ffw = FeedForward(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = self.ln_2(x)
        out = x + self.ffw(x)
        return out


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

        # Final layer norm + LM head
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

        # Pass through the decoder blocks
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        # LM head
        logits = self.linear(x)  # (b, t, n_embd) --> (b, t, vocab_size)

        if targets is not None:
            # Training
            logits = logits.view(-1, self.config.vocab_size)  # (b * t, vocab_size)
            targets = targets.view(-1)  # (b * t)
            loss = F.cross_entropy(logits, targets)
        else:
            # Inference
            loss = None

        return logits, loss
