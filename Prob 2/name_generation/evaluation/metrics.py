def compute_novelty(generated_names, training_names):

    # Normalize generated names to lowercase for fair comparison
    novel_names = [
        name for name in generated_names if name.strip().lower() not in training_names
    ]
    novelty_rate = len(novel_names) / len(generated_names) if generated_names else 0.0
    return novelty_rate, novel_names


def compute_diversity(generated_names):

    # Normalize to lowercase so "Aarav" and "aarav" count as the same
    normalized = [name.strip().lower() for name in generated_names]
    unique_names = set(normalized)
    diversity = len(unique_names) / len(normalized) if normalized else 0.0
    return diversity, unique_names
