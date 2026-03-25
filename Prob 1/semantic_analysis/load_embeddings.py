import os
import pickle
import numpy as np


def load_embeddings(pkl_path):

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    cbow_embeddings = {}
    sgns_embeddings = {}

    for key, vec in raw.items():
        if key.startswith("cbow_"):
            word = key[5:]  # strip "cbow_" prefix
            cbow_embeddings[word] = np.array(vec, dtype=np.float32)
        elif key.startswith("sgns_"):
            word = key[5:]  # strip "sgns_" prefix
            sgns_embeddings[word] = np.array(vec, dtype=np.float32)

    return cbow_embeddings, sgns_embeddings


def build_embedding_structures(word_vectors):
    # Alphabetical order for reproducibility
    sorted_words = sorted(word_vectors.keys())

    word_to_index = {w: i for i, w in enumerate(sorted_words)}
    index_to_word = {i: w for w, i in word_to_index.items()}

    # Stack all vectors into a 2-D matrix (each row is one word)
    embedding_matrix = np.stack([word_vectors[w] for w in sorted_words], axis=0)

    return word_to_index, index_to_word, embedding_matrix, word_vectors
