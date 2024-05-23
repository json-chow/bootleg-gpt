'''
Script for training the model
'''
import os
import torch
import configparser
import numpy as np
from tokenizer import BPETokenizer
from bootleg_gpt import BootlegGPTConfig, BootlegGPT


# Use GPU if available, otherwise use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

config = configparser.ConfigParser()
config.read("model.conf")

# Load data
context_length = config["model"]["n_ctx"]
batch_size = config["training"]["batch_size"]
tokenizer = BPETokenizer(config["tokenizer"]["vocab_file"], config["tokenizer"]["merges_file"])
data_loc = os.path.join(os.getcwd(), config["training"]["data_loc"])

# Tokenize data if only raw text exists (.raw --> .bin)
for file in os.listdir(data_loc):
    if file.endswith(".raw"):
        if not os.path.exists(os.path.join(data_loc, file.removesuffix(".raw") + ".bin")):
            pass
# def get_batch():
#     # Sample random indices
#     idxs = torch.randint(len(train_data) - context_length, (batch_size,))
#     x = torch.stack([train_data[idx:idx+context_length] for idx in idxs]).to(device)
#     # Get the next words for each instance of x
#     y = torch.stack([train_data[idx+1:idx+context_length+1] for idx in idxs]).to(device)
#     return x, y