import configparser
import os
import torch
from bootleg_gpt import BootlegGPTConfig, BootlegGPT
from tokenizer import BPETokenizer

'''
Script for actually running the model
'''
config = configparser.ConfigParser()
config.read("model.conf")
m_conf = config["model"]
tr_conf = config["training"]
tok_conf = config["tokenizer"]
out_conf = config["output"]

# Use GPU if available, otherwise use the CPU
device = m_conf["device"] if m_conf["device"] != "None" else "cuda" if torch.cuda.is_available() else "cpu"

# Check if model already exists in path, if so then load it
out_dir = tr_conf["out_dir"]
out_loc = os.path.join(out_dir, tr_conf["out_name"] + ".pt")
if os.path.exists(out_loc):
    # Load model and init from ckpt if model exists
    print("Loading existing model from", out_loc)
    ckpt = torch.load(out_loc, map_location=device)
    model_config = BootlegGPTConfig(**ckpt["model_conf"])
    model = BootlegGPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
else:
    raise FileNotFoundError("Specified model does not exist")
model.eval()

# Read/write from text file as specified in config
# Very crude interface for performing inference -- TODO: streamlit/some gui
tokenizer = BPETokenizer(tok_conf["vocab_file"], tok_conf["merges_file"])
txt_out_loc = out_conf["out_txt"]
f = open(txt_out_loc, "r+", encoding="utf-8")
with torch.no_grad():
    while True:
        curr_contents = f.read()
        print(f"Text: {curr_contents}")
        inp = None
        while inp not in {"g", "e"}:
            print("g: generate a token, e: exit loop", end=" ")
            inp = input()
        if inp == "g":
            tokens = torch.stack([tokenizer(curr_contents)]).to(device)
            # # greedy
            # out_token = tokenizer.decode(torch.argmax(model(tokens)[0][:, -1, :]).unsqueeze(0))
            # sampling distribution
            model_output = model(tokens)[0]
            out_token = torch.nn.functional.softmax(model_output[:,-1,:])  # final token in time 
            out_token = tokenizer.decode(torch.multinomial(out_token, 1).item())
            curr_contents += out_token
            f.write(out_token)
            f.seek(0)
        else:
            break
f.close()
