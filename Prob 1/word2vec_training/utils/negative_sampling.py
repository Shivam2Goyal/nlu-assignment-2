import random
import torch


class NegativeSampler:
    # size of the pre-built alias table (larger → less repetition)
    TABLE_SIZE = 1_000_000

    def __init__(self, unigram_probs, vocab_size):

        self.vocab_size = vocab_size
        self.table = self._build_table(unigram_probs)

    def _build_table(self, probs):

        table = []
        cumulative = 0.0
        word_idx = 0

        for i in range(self.TABLE_SIZE):
            # advance through the probability mass until we've
            # "used up" the current word's share of the table
            target_fraction = (i + 0.5) / self.TABLE_SIZE
            while cumulative < target_fraction and word_idx < len(probs):
                cumulative += probs[word_idx]
                word_idx += 1
            # word_idx was incremented past the match, step back
            table.append(max(word_idx - 1, 0))

        return table

    def sample(self, count, exclude_idx=None):

        negatives = []
        while len(negatives) < count:
            pos = random.randint(0, self.TABLE_SIZE - 1)
            word_id = self.table[pos]
            # avoid accidentally sampling the true positive word
            if word_id != exclude_idx:
                negatives.append(word_id)
        return negatives

    def sample_batch(self, batch_size, num_negatives, exclude_indices):
        
        excl = exclude_indices.tolist()
        rows = []
        for i in range(batch_size):
            rows.append(self.sample(num_negatives, exclude_idx=excl[i]))
        return torch.tensor(rows, dtype=torch.long)
