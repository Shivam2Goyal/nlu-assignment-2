from evaluation.metrics import compute_novelty, compute_diversity
from evaluation.generate_names import generate_names, save_generated_names


def evaluate_model(
    model,
    model_name,
    vocab,
    training_names_set,
    num_generate=1000,
    temperature=0.8,
    save_path=None,
):
    # Generate names from the model
    generated = generate_names(
        model, vocab, num_names=num_generate, temperature=temperature
    )

    # Save generated names to file if path provided
    if save_path:
        save_generated_names(generated, save_path)

    # Compute evaluation metrics
    novelty_rate, novel_names = compute_novelty(generated, training_names_set)
    diversity, unique_names = compute_diversity(generated)

    results = {
        "model_name": model_name,
        "num_generated": len(generated),
        "novelty_rate": novelty_rate,
        "num_novel": len(novel_names),
        "diversity": diversity,
        "num_unique": len(unique_names),
        "generated_names": generated,
        "novel_names": novel_names,
    }

    return results
