import pickle
import numpy as np


# Load the combined pickle and split by model prefix.
def load_embeddings(pkl_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    cbow_vectors = {}
    sgns_vectors = {}

    for key, vec in raw.items():
        if key.startswith("cbow_"):
            cbow_vectors[key[5:]] = np.array(vec, dtype=np.float32)
        elif key.startswith("sgns_"):
            sgns_vectors[key[5:]] = np.array(vec, dtype=np.float32)

    return cbow_vectors, sgns_vectors


# Build a matrix and filtered word list from the vocabulary subset that actually exists in the embeddings.
def extract_subset(word_vectors, word_list):
    words = [w for w in word_list if w in word_vectors]
    matrix = np.stack([word_vectors[w] for w in words], axis=0)
    return matrix, words
