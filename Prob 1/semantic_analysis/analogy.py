import numpy as np
from similarity import cosine_similarity

def solve_analogy(a, b, c, word_to_vector, word_to_index, embedding_matrix, top_k=5):

    # Check that all three words exist in the vocabulary
    for word in [a, b, c]:
        if word not in word_to_vector:
            return None

    # Vector arithmetic: the relationship from A to B, applied to C
    target_vector = word_to_vector[b] - word_to_vector[a] + word_to_vector[c]

    # Vectorised cosine similarity against the full vocabulary
    dots = embedding_matrix @ target_vector
    norms = np.linalg.norm(embedding_matrix, axis=1)
    target_norm = np.linalg.norm(target_vector)

    denom = norms * target_norm
    denom[denom == 0.0] = 1.0

    similarities = dots / denom

    # Sort by descending similarity
    sorted_indices = np.argsort(-similarities)

    # Exclude the three input words from the result
    exclude = {a, b, c}
    index_to_word = {i: w for w, i in word_to_index.items()}

    results = []
    for idx in sorted_indices:
        word = index_to_word[int(idx)]
        if word in exclude:
            continue
        results.append((word, float(similarities[idx])))
        if len(results) == top_k:
            break

    return results
