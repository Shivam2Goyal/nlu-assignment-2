import torch
import sys
import os

# Ensure parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.vanilla_rnn import VanillaRNN
from models.blstm import BLSTM
from models.attention_rnn import AttentionRNN


def load_model(
    model_class,
    vocab_size,
    embedding_dim,
    hidden_size,
    num_layers,
    model_path,
    device="cpu",
):
    
    model = model_class(vocab_size, embedding_dim, hidden_size, num_layers)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


def generate_names(model, vocab, num_names=1000, temperature=0.8, max_len=20):
    
    names = []
    for _ in range(num_names):
        with torch.no_grad():
            name = model.generate(vocab, temperature=temperature, max_len=max_len)
        # Only keep non-empty names
        if name.strip():
            names.append(name)
    return names


def save_generated_names(names, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
