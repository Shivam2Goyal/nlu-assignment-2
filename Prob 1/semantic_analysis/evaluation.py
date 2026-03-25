from similarity import get_nearest_neighbors
from analogy import solve_analogy


# Words required by the assignment 
REQUIRED_WORDS = ["research", "student", "phd", "exam"]

# Analogy experiments (A : B :: C : ?) 
# Each tuple is (A, B, C, description)
ANALOGY_EXPERIMENTS = [
    ("ug", "btech", "pg", "UG : BTech :: PG : ?"),
    ("faculty", "teaching", "student", "Faculty : Teaching :: Student : ?"),
    ("phd", "thesis", "mtech", "PhD : Thesis :: MTech : ?"),
    ("admission", "requirement", "registration", "Admission : Requirement :: Registration : ?"),
]


def run_neighbor_evaluation(word_to_vector, word_to_index, embedding_matrix, top_k=5):

    # Compute nearest neighbors for every required word.
    results = []
    for word in REQUIRED_WORDS:
        neighbors = get_nearest_neighbors(
            word, word_to_vector, word_to_index, embedding_matrix, top_k
        )
        results.append({"word": word, "neighbors": neighbors})
    return results


def run_analogy_evaluation(word_to_vector, word_to_index, embedding_matrix, top_k=5):

    results = []
    for a, b, c, desc in ANALOGY_EXPERIMENTS:
        predictions = solve_analogy(
            a, b, c, word_to_vector, word_to_index, embedding_matrix, top_k
        )
        results.append(
            {
                "description": desc,
                "a": a,
                "b": b,
                "c": c,
                "predictions": predictions,
            }
        )
    return results


def run_semantic_analysis(
    model_name, word_to_vector, word_to_index, embedding_matrix, top_k=5
):

    neighbor_results = run_neighbor_evaluation(
        word_to_vector, word_to_index, embedding_matrix, top_k
    )
    analogy_results = run_analogy_evaluation(
        word_to_vector, word_to_index, embedding_matrix, top_k
    )
    return {
        "model_name": model_name,
        "neighbor_results": neighbor_results,
        "analogy_results": analogy_results,
    }
