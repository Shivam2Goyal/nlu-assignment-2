import random
import torch
from torch.utils.data import Dataset

# Pair generation helpers
def generate_cbow_pairs(sentences, word_to_index, window_size=2):

    pairs = []
    for sent in sentences:
        # convert the sentence to index form, keeping only known words
        indexed = [word_to_index[w] for w in sent if w in word_to_index]
        length = len(indexed)

        # need at least 3 tokens to form a useful context–target pair
        if length < 3:
            continue

        for center_pos in range(length):
            target_idx = indexed[center_pos]

            # collect context word indices within the window
            context_ids = []
            for offset in range(-window_size, window_size + 1):
                neighbor_pos = center_pos + offset
                # skip the center word itself and out-of-bounds positions
                if offset == 0 or neighbor_pos < 0 or neighbor_pos >= length:
                    continue
                context_ids.append(indexed[neighbor_pos])

            # only keep samples where we got at least one context word
            if context_ids:
                pairs.append((context_ids, target_idx))

    return pairs


def generate_skipgram_pairs(sentences, word_to_index, window_size=2):

    pairs = []
    for sent in sentences:
        indexed = [word_to_index[w] for w in sent if w in word_to_index]
        length = len(indexed)

        if length < 2:
            continue

        for center_pos in range(length):
            center_idx = indexed[center_pos]

            for offset in range(-window_size, window_size + 1):
                neighbor_pos = center_pos + offset
                if offset == 0 or neighbor_pos < 0 or neighbor_pos >= length:
                    continue
                context_idx = indexed[neighbor_pos]
                pairs.append((center_idx, context_idx))

    return pairs


# PyTorch Dataset wrappers
class CBOWDataset(Dataset):
    # Wraps the CBOW (context, target) pairs into a PyTorch Dataset so we can use a DataLoader for batching and shuffling.

    def __init__(self, pairs, max_context_len):
        self.targets = []
        self.contexts = []
        self.masks = []

        for ctx_ids, tgt_id in pairs:
            # pad context to fixed length with zeros (index 0 is a real word,
            # but the mask will tell the model to ignore the padding slots)
            padded = ctx_ids[:max_context_len]  # truncate if needed
            mask = [1] * len(padded)
            while len(padded) < max_context_len:
                padded.append(0)
                mask.append(0)

            self.contexts.append(padded)
            self.masks.append(mask)
            self.targets.append(tgt_id)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.contexts[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


class SkipGramDataset(Dataset): 

    def __init__(self, pairs):
        self.centers = [p[0] for p in pairs]
        self.contexts = [p[1] for p in pairs]

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.centers[idx], dtype=torch.long),
            torch.tensor(self.contexts[idx], dtype=torch.long),
        )
