import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # Center-word embeddings (the ones we ultimately keep)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Context-word embeddings (used during training, then discarded)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Small uniform initialisation (like the original C code)
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, center_ids, context_ids, negative_ids):

        # Embedding lookups
        # center vectors: (batch, emb_dim)
        v_center = self.input_embeddings(center_ids)
        # positive context vectors: (batch, emb_dim)
        v_context = self.output_embeddings(context_ids)
        # negative vectors: (batch, K, emb_dim)
        v_negatives = self.output_embeddings(negative_ids)

        # Positive score
        # Dot product between center and true context for each sample.
        pos_dot = (v_center * v_context).sum(dim=1)
        pos_loss = F.logsigmoid(pos_dot)  # log σ(v_c · v_o)

        # Negative scores
        # For each negative word, dot product with center should be
        # small (sigmoid of the *negated* dot product should be high).
        neg_dot = torch.bmm(v_negatives, v_center.unsqueeze(2)).squeeze(2)  # (batch, K)
        neg_loss = F.logsigmoid(-neg_dot).sum(dim=1)  # sum over K

        # Total loss
        # We want to *maximise* (pos_loss + neg_loss), so we
        # *minimise* the negation.
        loss = -(pos_loss + neg_loss).mean()

        return loss

    def get_embeddings(self):
        return self.input_embeddings.weight.data.clone()
