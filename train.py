import os
import torch
import configparser
import mmap
import numpy as np
from tqdm import tqdm
from tokenizer import BPETokenizer
from bootleg_gpt import BootlegGPTConfig, BootlegGPT
from torchinfo import summary


'''
Script for training the model
'''
config = configparser.ConfigParser()
config.read("model.conf")
m_conf = config["model"]
tr_conf = config["training"]
tok_conf = config["tokenizer"]

# Use GPU if available, otherwise use the CPU
device = m_conf["device"] if m_conf["device"] != "None" else "cuda" if torch.cuda.is_available() else "cpu"

context_length = int(m_conf["n_ctx"])
batch_size = int(tr_conf["batch_size"])
tokenizer = BPETokenizer(tok_conf["vocab_file"], tok_conf["merges_file"])
data_loc = os.path.join(os.getcwd(), tr_conf["data_loc"])

# Tokenize data if only raw text exists (.raw --> .bin)
for file in os.listdir(data_loc):
    if file.endswith(".raw"):
        if not os.path.exists(os.path.join(data_loc, file.removesuffix(".raw") + ".bin")):
            # Open raw text file and tokenize the text
            with open(os.path.join(data_loc, file), encoding="utf-8") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                tokens = tokenizer(mm, return_tensors=True)
                mm.close()
            # Write the tokens to a bin file
            with open(os.path.join(data_loc, file.removesuffix(".raw") + ".bin"), "wb") as f:
                np.save(f, tokens.numpy())

# Load data
train_data = np.load(os.path.join(data_loc, "train.bin"), mmap_mode="r")
val_data = np.load(os.path.join(data_loc, "val.bin"), mmap_mode="r")
def get_batch(split):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise Exception("No split of such name")
    # Sample random indices
    idxs = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[idx:idx+context_length]) for idx in idxs]).to(device)
    # Get the next words for each instance of x
    y = torch.stack([torch.from_numpy(data[idx+1:idx+context_length+1]) for idx in idxs]).to(device)
    return x, y

# Check if model already exists in path, if so then load it
out_dir = tr_conf["out_dir"]
out_loc = os.path.join(out_dir, tr_conf["out_name"] + ".pt")
if os.path.exists(out_loc):
    # Load model and init from ckpt if model exists
    print("Loading existing model...")
    ckpt = torch.load(out_loc)
    ckpt["model_conf"]["device"] = device
    model_config = BootlegGPTConfig(**ckpt["model_conf"])
    model = BootlegGPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr_conf["lr"]), weight_decay=float(tr_conf["weight_decay"]))
    optimizer.load_state_dict(ckpt["optim_state_dict"])
    iter_num = ckpt["iter_num"]
    train_losses = ckpt["train_losses"]
    val_losses = ckpt["val_losses"]
else:
    # Initialize the model
    print("Initializing the model...")
    model_config = BootlegGPTConfig(
        vocab_size=tokenizer.vocab_len,
        n_ctx=context_length,
        n_embd=int(m_conf["n_embd"]),
        n_layer=int(m_conf["n_layer"]),
        n_head=int(m_conf["n_head"]),
        n_inner=None if m_conf["n_inner"] == "None" else int(m_conf["n_inner"]),
        drop=float(m_conf["drop"]),
        layer_norm_eps=float(m_conf["layer_norm_eps"]),
        bias=m_conf.getboolean("bias"),
        device=device
    )
    model = BootlegGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr_conf["lr"]), weight_decay=float(tr_conf["weight_decay"]))
    iter_num = 0
    train_losses = []
    val_losses = []
summary(model, input_data=get_batch("train"), device=device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_iters = int(tr_conf["eval_iters"])
    with tqdm(total=eval_iters*2, desc=f"Estimating loss") as pbar:
        for split in ["train", "val"]:
            avg_loss = 0
            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                avg_loss += loss
                pbar.update(1)
            out[split] = avg_loss / eval_iters
    model.train()
    return out

# Training loop
max_iters = int(tr_conf["max_iters"])
with tqdm(total=max_iters, initial=iter_num, desc="Training model") as pbar:
    while iter_num <= max_iters :
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1
        pbar.update(1)
        # Write ckpt every 1000 iters (for now)
        if ((iter_num % 1000 == 0) or (iter_num == max_iters)):
            losses = estimate_loss()
            train_losses.append(losses["train"].item())
            val_losses.append(losses["val"].item())
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "model_conf": model_config.__dict__,
                "iter_num": iter_num,
                "train_losses": train_losses,
                "val_losses": val_losses
            }
            print(f"\nSaving checkpoint, Step: {iter_num}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
            torch.save(ckpt, os.path.join(out_loc))
