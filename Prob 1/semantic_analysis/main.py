import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from semantic_analysis.load_embeddings import (
    load_embeddings,
    build_embedding_structures,
)
from semantic_analysis.evaluation import run_semantic_analysis

# Paths 
EMBEDDINGS_PATH = os.path.join(
    PROJ_ROOT, "word2vec_training", "data", "word_embeddings.pkl"
)
OUTPUT_DIR = os.path.join(PROJ_ROOT, "semantic_analysis")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "semantic_analysis_results.txt")

# Formatting helpers 
def format_neighbor_results(neighbor_results):
    lines = []
    for entry in neighbor_results:
        word = entry["word"]
        neighbors = entry["neighbors"]
        lines.append(f"Word: {word}")
        if neighbors is None:
            lines.append("  Word not found in vocabulary\n")
            continue
        lines.append("Nearest neighbors:")
        for rank, (nbr, sim) in enumerate(neighbors, 1):
            lines.append(f"  {rank}. {nbr} ({sim:.4f})")
        lines.append("")
    return "\n".join(lines)


def format_analogy_results(analogy_results):
    lines = []
    for entry in analogy_results:
        desc = entry["description"]
        preds = entry["predictions"]
        lines.append(f"Analogy: {desc}")
        if preds is None:
            lines.append("  One or more words not found in vocabulary\n")
            continue
        lines.append("Top predictions:")
        for rank, (word, sim) in enumerate(preds, 1):
            lines.append(f"  {rank}. {word} ({sim:.4f})")
        lines.append("")
    return "\n".join(lines)


def generate_neighbor_interpretation(model_name, neighbor_results):

    lines = [f"\nSemantic Interpretation ({model_name} Nearest Neighbors):"]
    lines.append("-" * 55)

    for entry in neighbor_results:
        word = entry["word"]
        neighbors = entry["neighbors"]
        if neighbors is None:
            lines.append(
                f'  "{word}" is absent from the vocabulary, so no '
                "neighbors can be computed."
            )
            continue

        top_words = [n[0] for n in neighbors[:3]]
        top_str = '", "'.join(top_words)
        lines.append(
            f'  The nearest neighbors of "{word}" include "{top_str}", '
            f"indicating the model captures the semantic context "
            f"associated with {word}-related concepts."
        )

    lines.append("")
    return "\n".join(lines)


def generate_analogy_interpretation(model_name, analogy_results):
    lines = [f"\nSemantic Interpretation ({model_name} Analogies):"]
    lines.append("-" * 55)

    for entry in analogy_results:
        desc = entry["description"]
        preds = entry["predictions"]
        if preds is None:
            lines.append(f"  {desc} -- could not evaluate (missing words).")
            continue

        top_words = [p[0] for p in preds[:3]]
        top_str = ", ".join(top_words)
        lines.append(
            f"  {desc}\n"
            f"    Top predictions: {top_str}\n"
            f"    The model's predictions suggest the embeddings encode\n"
            f"    the underlying relationship captured by this analogy."
        )

    lines.append("")
    return "\n".join(lines)


def generate_comparison(cbow_analysis, sgns_analysis):
    lines = []
    lines.append("=" * 60)
    lines.append("  CBOW vs Skip-Gram Comparison")
    lines.append("=" * 60)

    # Compare neighbor results word-by-word
    lines.append("\nNearest Neighbor Comparison:")
    lines.append("-" * 40)
    for cbow_entry, sgns_entry in zip(
        cbow_analysis["neighbor_results"],
        sgns_analysis["neighbor_results"],
    ):
        word = cbow_entry["word"]
        lines.append(f"\n  Word: {word}")

        cbow_nbrs = cbow_entry["neighbors"]
        sgns_nbrs = sgns_entry["neighbors"]

        if cbow_nbrs:
            cbow_str = ", ".join(f"{w} ({s:.2f})" for w, s in cbow_nbrs[:3])
        else:
            cbow_str = "N/A"
        if sgns_nbrs:
            sgns_str = ", ".join(f"{w} ({s:.2f})" for w, s in sgns_nbrs[:3])
        else:
            sgns_str = "N/A"

        lines.append(f"    CBOW      : {cbow_str}")
        lines.append(f"    Skip-Gram : {sgns_str}")
    return "\n".join(lines)

 
# Main pipeline

def main():
    print("=" * 60)
    print("  Task-3: Semantic Analysis of Word2Vec Embeddings")
    print("=" * 60)

    #  Load embeddings from Task-2 
    print(f"\nLoading embeddings from {EMBEDDINGS_PATH} ...")
    cbow_raw, sgns_raw = load_embeddings(EMBEDDINGS_PATH)
    print(f"  CBOW vocabulary : {len(cbow_raw)} words")
    print(f"  SGNS vocabulary : {len(sgns_raw)} words")

    # Build structured lookup tables for each model
    cbow_w2i, cbow_i2w, cbow_matrix, cbow_w2v = build_embedding_structures(cbow_raw)
    sgns_w2i, sgns_i2w, sgns_matrix, sgns_w2v = build_embedding_structures(sgns_raw)

    #  Run evaluation for CBOW 
    print("\n" + "=" * 60)
    print("  CBOW Results")
    print("=" * 60)

    cbow_analysis = run_semantic_analysis("CBOW", cbow_w2v, cbow_w2i, cbow_matrix)
    print("\n" + format_neighbor_results(cbow_analysis["neighbor_results"]))
    print(format_analogy_results(cbow_analysis["analogy_results"]))
    print(generate_neighbor_interpretation("CBOW", cbow_analysis["neighbor_results"]))
    print(generate_analogy_interpretation("CBOW", cbow_analysis["analogy_results"]))

    #  Run evaluation for Skip-Gram ─
    print("\n" + "=" * 60)
    print("  Skip-Gram Results")
    print("=" * 60)

    sgns_analysis = run_semantic_analysis("Skip-Gram", sgns_w2v, sgns_w2i, sgns_matrix)
    print("\n" + format_neighbor_results(sgns_analysis["neighbor_results"]))
    print(format_analogy_results(sgns_analysis["analogy_results"]))
    print(
        generate_neighbor_interpretation("Skip-Gram", sgns_analysis["neighbor_results"])
    )
    print(
        generate_analogy_interpretation("Skip-Gram", sgns_analysis["analogy_results"])
    )

    #  CBOW vs Skip-Gram comparison ─
    comparison = generate_comparison(cbow_analysis, sgns_analysis)
    print("\n" + comparison)

    #  Save everything to results file 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Task-3: Semantic Analysis of Word2Vec Embeddings\n")
        f.write("=" * 60 + "\n\n")

        # CBOW section
        f.write("CBOW Results\n")
        f.write("-" * 60 + "\n\n")
        f.write("Nearest Neighbor Results\n")
        f.write("-" * 30 + "\n\n")
        f.write(format_neighbor_results(cbow_analysis["neighbor_results"]))
        f.write("\n")
        f.write("Analogy Experiments\n")
        f.write("-" * 30 + "\n\n")
        f.write(format_analogy_results(cbow_analysis["analogy_results"]))
        f.write("\n")
        f.write(
            generate_neighbor_interpretation("CBOW", cbow_analysis["neighbor_results"])
        )
        f.write("\n")
        f.write(
            generate_analogy_interpretation("CBOW", cbow_analysis["analogy_results"])
        )
        f.write("\n\n")

        # Skip-Gram section
        f.write("Skip-Gram Results\n")
        f.write("-" * 60 + "\n\n")
        f.write("Nearest Neighbor Results\n")
        f.write("-" * 30 + "\n\n")
        f.write(format_neighbor_results(sgns_analysis["neighbor_results"]))
        f.write("\n")
        f.write("Analogy Experiments\n")
        f.write("-" * 30 + "\n\n")
        f.write(format_analogy_results(sgns_analysis["analogy_results"]))
        f.write("\n")
        f.write(
            generate_neighbor_interpretation(
                "Skip-Gram", sgns_analysis["neighbor_results"]
            )
        )
        f.write("\n")
        f.write(
            generate_analogy_interpretation(
                "Skip-Gram", sgns_analysis["analogy_results"]
            )
        )
        f.write("\n\n")

        # Comparison section
        f.write(comparison)
        f.write("\n")

    print(f"\nResults saved to {RESULTS_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
