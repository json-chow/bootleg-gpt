import torch
import os
import configparser
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read("model.conf")
tr_conf = config["training"]
m_conf = config["model"]

device = m_conf["device"] if m_conf["device"] != "None" else "cuda" if torch.cuda.is_available() else "cpu"

out_dir = tr_conf["out_dir"]
out_loc = os.path.join(out_dir, tr_conf["out_name"] + ".pt")
if os.path.exists(out_loc):
    # Load model and init from ckpt if model exists
    print("Loading existing model...")
    ckpt = torch.load(out_loc, map_location=device)
    train_losses = ckpt["train_losses"]
    val_losses = ckpt["val_losses"]
    iter_num = ckpt["iter_num"]

xs = [i for i in range(0, iter_num+1, iter_num//len(train_losses))][1:]
plt.plot(xs, [i.item() if type(i) == torch.Tensor else i for i in train_losses], label="Train Loss")
plt.plot(xs, [i.item() if type(i) == torch.Tensor else i for i in val_losses], label="Val Loss")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()