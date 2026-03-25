from sklearn.manifold import TSNE
import numpy as np


def apply_tsne(embedding_matrix, n_components=2, perplexity=30, random_state=42):
    
    # Perplexity must be less than N; clamp if our subset is small
    effective_perplexity = min(perplexity, embedding_matrix.shape[0] - 1)

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    reduced = tsne.fit_transform(embedding_matrix)
    return reduced
