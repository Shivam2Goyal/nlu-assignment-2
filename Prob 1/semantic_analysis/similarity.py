import numpy as np


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    # Guard against zero-vector edge case (would cause division by zero)
    if norm_product == 0.0:
        return 0.0

    return float(dot_product / norm_product)


def get_nearest_neighbors(
    word, word_to_vector, word_to_index, embedding_matrix, top_k=5
):
    if word not in word_to_vector:
        return None

    target_vec = word_to_vector[word]

    # Vectorised cosine similarity against every row of the matrix:
    #   cos(target, row_i) = target . row_i / (||target|| * ||row_i||)
    # Compute dot products with all words at once
    dots = embedding_matrix @ target_vec  # (vocab_size,)
    # L2 norms for every word vector (row-wise)
    norms = np.linalg.norm(embedding_matrix, axis=1)  # (vocab_size,)
    target_norm = np.linalg.norm(target_vec)

    # Avoid division by zero for any zero-norm row
    denom = norms * target_norm
    denom[denom == 0.0] = 1.0

    similarities = dots / denom  # (vocab_size,)

    # Sort indices by similarity in descending order
    sorted_indices = np.argsort(-similarities)

    # Collect top_k results, skipping the query word itself
    index_to_word = {i: w for w, i in word_to_index.items()}
    neighbors = []
    for idx in sorted_indices:
        neighbor_word = index_to_word[int(idx)]
        if neighbor_word == word:
            continue
        neighbors.append((neighbor_word, float(similarities[idx])))
        if len(neighbors) == top_k:
            break

    return neighbors
