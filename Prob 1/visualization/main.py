import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from load_embeddings import load_embeddings, extract_subset
from pca_visualization import apply_pca
from tsne_visualization import apply_tsne
from plot_utils import plot_embeddings, SEMANTIC_GROUPS

# Paths
EMBEDDINGS_PATH = os.path.join(
    PROJ_ROOT, "word2vec_training", "data", "word_embeddings.pkl"
)
VIS_DIR = os.path.join(PROJ_ROOT, "visualization", "visualizations")

# Words to visualize
# A curated subset of ~44 words spanning five semantic categories
# relevant to the IIT Jodhpur corpus.  Plotting all ~1900 words
# would produce an unreadable figure; a focused subset lets us
# clearly see whether the embeddings form meaningful clusters.
VISUALIZATION_WORDS = [
    # Degree programs
    "btech",
    "mtech",
    "phd",
    "mba",
    "ug",
    "pg",
    # Academic / student life
    "student",
    "faculty",
    "exam",
    "course",
    "program",
    "department",
    "semester",
    "academic",
    "curriculum",
    "professor",
    "dean",
    "director",
    # Research & innovation
    "research",
    "laboratory",
    "project",
    "innovation",
    "technology",
    "development",
    "conference",
    "publication",
    "journal",
    "workshop",
    # Campus & institution
    "iit",
    "jodhpur",
    "campus",
    "institute",
    "education",
    "library",
    "sports",
    "cultural",
    "committee",
    "council",
    # Career & training
    "engineering",
    "science",
    "knowledge",
    "training",
    "scholarship",
    "placement",
]


def generate_cluster_interpretation(model_name, reduced, words):
    import numpy as np

    # Build group membership for words that are in a known group
    word_groups = {}
    for group, members in SEMANTIC_GROUPS.items():
        for w in members:
            if w in words:
                word_groups[w] = group

    # Intra-group average distance (tightness)
    group_tightness = {}
    for group in SEMANTIC_GROUPS:
        idxs = [i for i, w in enumerate(words) if word_groups.get(w) == group]
        if len(idxs) < 2:
            continue
        pts = reduced[idxs]
        # Mean pairwise Euclidean distance within the group
        dists = []
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dists.append(np.linalg.norm(pts[a] - pts[b]))
        group_tightness[group] = np.mean(dists)

    # Overall centroid distance between groups (separation)
    centroids = {}
    for group in SEMANTIC_GROUPS:
        idxs = [i for i, w in enumerate(words) if word_groups.get(w) == group]
        if idxs:
            centroids[group] = reduced[idxs].mean(axis=0)

    lines = []
    lines.append(f"\nCluster Analysis ({model_name}):")
    lines.append("-" * 50)

    lines.append("\n  Intra-group tightness (lower = tighter cluster):")
    for group, tight in sorted(group_tightness.items(), key=lambda x: x[1]):
        lines.append(f"    {group:<28s}  avg dist = {tight:.2f}")

    if len(centroids) >= 2:
        lines.append("\n  Inter-group centroid distances:")
        groups_list = sorted(centroids.keys())
        for i in range(len(groups_list)):
            for j in range(i + 1, len(groups_list)):
                d = np.linalg.norm(
                    centroids[groups_list[i]] - centroids[groups_list[j]]
                )
                lines.append(
                    f"    {groups_list[i]:<28s} <-> "
                    f"{groups_list[j]:<28s}  dist = {d:.2f}"
                )

    return "\n".join(lines)


def main():
    os.makedirs(VIS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Task-4: Word Embedding Visualization")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {EMBEDDINGS_PATH} ...")
    cbow_raw, sgns_raw = load_embeddings(EMBEDDINGS_PATH)
    print(f"  CBOW vocabulary : {len(cbow_raw)} words")
    print(f"  SGNS vocabulary : {len(sgns_raw)} words")

    # Extract the curated word subset for each model
    cbow_matrix, cbow_words = extract_subset(cbow_raw, VISUALIZATION_WORDS)
    sgns_matrix, sgns_words = extract_subset(sgns_raw, VISUALIZATION_WORDS)
    print(f"  Words selected for visualization: {len(cbow_words)}")
    missing = [w for w in VISUALIZATION_WORDS if w not in cbow_raw]
    if missing:
        print(f"  Words not in vocabulary (skipped): {missing}")

    # PCA projections
    # PCA identifies directions of maximum variance in the embedding
    # space and projects vectors onto those directions.  It shows the
    # global geometric structure of the embeddings.
    print("\nApplying PCA ...")
    cbow_pca, pca_obj_cbow = apply_pca(cbow_matrix)
    sgns_pca, pca_obj_sgns = apply_pca(sgns_matrix)

    var_cbow = pca_obj_cbow.explained_variance_ratio_
    var_sgns = pca_obj_sgns.explained_variance_ratio_
    print(f"  CBOW PCA explained variance: {var_cbow[0]:.2%}, {var_cbow[1]:.2%}")
    print(f"  SGNS PCA explained variance: {var_sgns[0]:.2%}, {var_sgns[1]:.2%}")

    # t-SNE projections
    # t-SNE preserves local neighborhood structure, making it
    # effective for visualizing tight semantic clusters among words.
    print("\nApplying t-SNE ...")
    cbow_tsne = apply_tsne(cbow_matrix)
    sgns_tsne = apply_tsne(sgns_matrix)

    # Generate all four plots
    print("\nGenerating plots ...")

    plot_embeddings(
        cbow_pca,
        cbow_words,
        "CBOW Embeddings - PCA Projection",
        os.path.join(VIS_DIR, "cbow_pca.png"),
    )
    plot_embeddings(
        cbow_tsne,
        cbow_words,
        "CBOW Embeddings - t-SNE Projection",
        os.path.join(VIS_DIR, "cbow_tsne.png"),
    )
    plot_embeddings(
        sgns_pca,
        sgns_words,
        "Skip-Gram Embeddings - PCA Projection",
        os.path.join(VIS_DIR, "skipgram_pca.png"),
    )
    plot_embeddings(
        sgns_tsne,
        sgns_words,
        "Skip-Gram Embeddings - t-SNE Projection",
        os.path.join(VIS_DIR, "skipgram_tsne.png"),
    )

    # Cluster analysis─
    cbow_pca_interp = generate_cluster_interpretation(
        "CBOW - PCA", cbow_pca, cbow_words
    )
    cbow_tsne_interp = generate_cluster_interpretation(
        "CBOW - t-SNE", cbow_tsne, cbow_words
    )
    sgns_pca_interp = generate_cluster_interpretation(
        "Skip-Gram - PCA", sgns_pca, sgns_words
    )
    sgns_tsne_interp = generate_cluster_interpretation(
        "Skip-Gram - t-SNE", sgns_tsne, sgns_words
    )

    print(cbow_pca_interp)
    print(cbow_tsne_interp)
    print(sgns_pca_interp)
    print(sgns_tsne_interp)

    # Save interpretation to text file─
    interp_path = os.path.join(VIS_DIR, "cluster_interpretation.txt")
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write("Task-4: Embedding Visualization -- Cluster Interpretation\n")
        f.write("=" * 60 + "\n\n")
        f.write(cbow_pca_interp + "\n\n")
        f.write(cbow_tsne_interp + "\n\n")
        f.write(sgns_pca_interp + "\n\n")
        f.write(sgns_tsne_interp + "\n\n")
    print(f"Interpretation saved to {interp_path}")

    print("\nDone. Generated files:")
    for fname in [
        "cbow_pca.png",
        "cbow_tsne.png",
        "skipgram_pca.png",
        "skipgram_tsne.png",
        "cluster_interpretation.txt",
    ]:
        print(f"  {os.path.join(VIS_DIR, fname)}")


if __name__ == "__main__":
    main()
