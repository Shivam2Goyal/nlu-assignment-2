import torch
import torch.nn as nn


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # The embedding layer maps each word index to a dense vector.
        # These are the vectors we ultimately want to learn — they
        # will become our "word embeddings".
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # A linear layer that projects the averaged context embedding
        # back to vocabulary-sized logits so we can apply softmax /
        # cross-entropy to predict the center word.
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Initialise weights with small random values to break symmetry
        # and help gradients flow evenly at the start of training.
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, context_indices, context_mask):

        # Step 1 — Look up embeddings: (batch, max_ctx, emb_dim)
        embedded = self.embeddings(context_indices)

        # Step 2 — Zero out the padded positions so they don't
        # contribute to the average.
        # Expand mask to match embedding dimensions:
        #   (batch, max_ctx, 1)  so broadcasting works.
        mask_expanded = context_mask.unsqueeze(-1)
        embedded = embedded * mask_expanded

        # Step 3 — Average the context embeddings.
        # Divide by the actual number of context words (not the
        # padded length) to get a true mean.
        num_real = context_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_embedding = embedded.sum(dim=1) / num_real  # (batch, emb_dim)

        # Step 4 — Project to vocabulary logits.
        logits = self.linear(avg_embedding)  # (batch, vocab_size)

        return logits

    def get_embeddings(self):
        return self.embeddings.weight.data.clone()
