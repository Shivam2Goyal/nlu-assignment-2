import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NameDataset(Dataset):

    def __init__(self, names, vocab):
        self.vocab = vocab
        self.names = names
        # Pre-encode all names into index sequences
        self.encoded = [
            torch.tensor(vocab.encode(name), dtype=torch.long) for name in names
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]
        # Input: all tokens except the last (<START> ... last_char)
        # Target: all tokens except the first (first_char ... <END>)
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq


def load_names(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def collate_fn(batch):
    inputs, targets = zip(*batch)
    # Pad sequences with 0 (PAD index)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded