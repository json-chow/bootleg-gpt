import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class BootlegGPTConfig():
    vocab_size: int = 50257  # Size of vocabulary
    block_size: int = 1024  # Max context length
    n_embd: int = 768  # Embedding size
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_inner: int = None  # Dimensionality of feedforward layers, default is 4 * n_embd
    drop: int = 0.1  # Dropout probability
    layer_norm_eps: int = 1e-5  # Epsilon for layer norm layers
    bias: bool = True  # Include bias in linear layers

    def __post_init__(self):
        if self.n_inner == None:
            self.n_inner = 4 * self.n_embd


class BootlegGPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config


if __name__ == "__main__":
    # Tests
    c = BootlegGPTConfig(n_embd=12)
    print(c)
