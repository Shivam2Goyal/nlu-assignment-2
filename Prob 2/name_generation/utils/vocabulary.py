class CharVocabulary:

    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    PAD_TOKEN = "<PAD>"

    def __init__(self):
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0

    def build(self, names):
        # Collect all unique characters across all names
        chars = set()
        for name in names:
            for ch in name:
                chars.add(ch)

        # Sort for reproducibility
        sorted_chars = sorted(chars)

        # Reserve indices for special tokens first
        special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
        all_tokens = special_tokens + sorted_chars

        self.char_to_index = {token: idx for idx, token in enumerate(all_tokens)}
        self.index_to_char = {idx: token for token, idx in self.char_to_index.items()}
        self.vocab_size = len(all_tokens)

        return self

    @property
    def start_idx(self):
        return self.char_to_index[self.START_TOKEN]

    @property
    def end_idx(self):
        return self.char_to_index[self.END_TOKEN]

    @property
    def pad_idx(self):
        return self.char_to_index[self.PAD_TOKEN]

    def encode(self, name):
        indices = [self.start_idx]
        for ch in name:
            indices.append(self.char_to_index[ch])
        indices.append(self.end_idx)
        return indices

    def decode(self, indices):
        chars = []
        for idx in indices:
            token = self.index_to_char[idx]
            if token == self.END_TOKEN:
                break
            if token == self.START_TOKEN or token == self.PAD_TOKEN:
                continue
            chars.append(token)
        return "".join(chars)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (
            f"CharVocabulary(vocab_size={self.vocab_size}, "
            f"characters={list(self.char_to_index.keys())})"
        )
