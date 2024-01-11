import torch
import torch.nn as nn
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

    def __init__(self):
        super().__init__()


class MultiHeadAttention(nn.Module):
    '''Concatenation of parallel scaled dot product attention layers'''

    def __init__(self):
        super().__init__()


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
        self.attn = 0
        self.ln_2 = LayerNorm(config)
        self.ffw = 0


# Based on the nanogpt implementation of gpt2
class BootlegGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Word token embedding / word position embedding lookup tables
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)

        # Dropout layer
        self.drop = nn.Dropout(c.drop)
    
    def forward(self, idx, target=None):
        # idxs come in with a batch and time dimension: b, t
        b, t = idx.size()

        # After passing idx to the embedding tables, channel dimension gets added (n_embd)
        tok_emb = self.wte(idx)  # (b, t) --> (b, t, n_embd)
        pos = torch.arange(t, device=idx.device)
        pos_emb = self.wpe(pos)  # (t) --> (t, n_embd)

        # Embedding dropout
        x = self.drop(tok_emb + pos_emb)
        y = x.mean(2, keepdim=True)
        
        return x


if __name__ == "__main__":
    # Tests
    c = BootlegGPTConfig(vocab_size=500, n_ctx=4, n_embd=8)
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
    test_x, test_y = get_batch()
    print(test_x)
    model = BootlegGPT(c).to(device)
    print(model(test_x))

