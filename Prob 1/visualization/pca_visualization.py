from sklearn.decomposition import PCA
import numpy as np


def apply_pca(embedding_matrix, n_components=2):
    
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embedding_matrix)
    return reduced, pca
