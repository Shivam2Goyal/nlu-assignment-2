import os
import sys

# Setup paths
# BASE_DIR points to name_generation/ (parent of evaluation/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
from utils.vocabulary import CharVocabulary
from utils.dataset import load_names
from models.vanilla_rnn import VanillaRNN
from models.blstm import BLSTM
from models.attention_rnn import AttentionRNN
from evaluation.generate_names import load_model
from evaluation.evaluation import evaluate_model

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "TrainingNames.txt")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters (must match Task-1 training)─
EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_GENERATE = 1000
TEMPERATURE = 0.8


def main():
    print("=" * 65)
    print("Task-2: Quantitative Evaluation of Name Generation Models")
    print("=" * 65)

    # Step 1: Load training dataset for novelty comparison─
    print("\n[Step 1] Loading training dataset...")
    names = load_names(DATA_PATH)
    # Normalize to lowercase and store as set for fast novelty lookup
    training_names_set = set(name.strip().lower() for name in names)
    print(f"  Loaded {len(names)} training names ({len(training_names_set)} unique)")

    # Build vocabulary (must match training)
    print("\n[Step 2] Rebuilding character vocabulary...")
    vocab = CharVocabulary().build(names)
    print(f"  Vocabulary size: {vocab.vocab_size}")

    # Create output directory
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Step 3: Load trained models (no retraining)─
    print("\n[Step 3] Loading trained models...")

    model_configs = [
        (
            "Vanilla RNN",
            VanillaRNN,
            os.path.join(SAVE_DIR, "vanilla_rnn.pt"),
            "rnn_generated.txt",
        ),
        ("BLSTM", BLSTM, os.path.join(SAVE_DIR, "blstm.pt"), "blstm_generated.txt"),
        (
            "RNN + Attention",
            AttentionRNN,
            os.path.join(SAVE_DIR, "attention_rnn.pt"),
            "attention_generated.txt",
        ),
    ]

    all_results = []

    for model_name, model_class, model_path, gen_filename in model_configs:
        print(f"\n{'─'*65}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*65}")

        # Load pre-trained model weights
        model = load_model(
            model_class,
            vocab.vocab_size,
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            NUM_LAYERS,
            model_path,
            DEVICE,
        )
        print(f"  Loaded weights from {os.path.basename(model_path)}")

        # Generate names and compute metrics
        gen_path = os.path.join(EVAL_DIR, gen_filename)
        results = evaluate_model(
            model,
            model_name,
            vocab,
            training_names_set,
            num_generate=NUM_GENERATE,
            temperature=TEMPERATURE,
            save_path=gen_path,
        )
        all_results.append(results)

        # Print per-model results
        print(f"  Generated {results['num_generated']} names")
        print(f"  Saved to: {gen_filename}")
        print(
            f"  Novelty Rate : {results['novelty_rate']:.4f} "
            f"({results['num_novel']} novel out of {results['num_generated']})"
        )
        print(
            f"  Diversity    : {results['diversity']:.4f} "
            f"({results['num_unique']} unique out of {results['num_generated']})"
        )

        # Show some sample generated names
        print(f"\n  Sample generated names (first 10):")
        for name in results["generated_names"][:10]:
            print(f"    {name}")

    # Step 7: Comparison Table
    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS")
    print("=" * 65)
    header = f"{'Model':<22} {'Novelty Rate':>14} {'Diversity':>12}"
    separator = "-" * 52
    print(header)
    print(separator)
    for r in all_results:
        print(
            f"{r['model_name']:<22} {r['novelty_rate']:>14.4f} {r['diversity']:>12.4f}"
        )
    print(separator)

    # Save results to text file─
    results_path = os.path.join(EVAL_DIR, "evaluation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("Task-2: Quantitative Evaluation Results\n")
        f.write("=" * 65 + "\n\n")

        f.write("Evaluation Settings:\n")
        f.write(f"  Names generated per model : {NUM_GENERATE}\n")
        f.write(f"  Sampling temperature      : {TEMPERATURE}\n")
        f.write(f"  Training set size         : {len(training_names_set)}\n")
        f.write(f"  Vocabulary size           : {vocab.vocab_size}\n\n")

        # Per-model details
        for r in all_results:
            f.write(f"{'─'*65}\n")
            f.write(f"Model: {r['model_name']}\n")
            f.write(f"{'─'*65}\n")
            f.write(f"  Generated names : {r['num_generated']}\n")
            f.write(
                f"  Novelty Rate    : {r['novelty_rate']:.4f} "
                f"({r['num_novel']} novel names)\n"
            )
            f.write(
                f"  Diversity       : {r['diversity']:.4f} "
                f"({r['num_unique']} unique names)\n"
            )
            f.write(f"\n  Sample generated names:\n")
            for name in r["generated_names"][:15]:
                f.write(f"    {name}\n")
            f.write(f"\n  Sample novel names:\n")
            for name in r["novel_names"][:15]:
                f.write(f"    {name}\n")
            f.write("\n")

        # Comparison table
        f.write("=" * 65 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Model':<22} {'Novelty Rate':>14} {'Diversity':>12}\n")
        f.write("-" * 52 + "\n")
        for r in all_results:
            f.write(
                f"{r['model_name']:<22} {r['novelty_rate']:>14.4f} "
                f"{r['diversity']:>12.4f}\n"
            )
        f.write("-" * 52 + "\n")

        # Observations / interpretation
        f.write("\n" + "=" * 65 + "\n")
        f.write("OBSERVATIONS\n")
        f.write("=" * 65 + "\n\n")

        # Sort models by novelty and diversity for interpretation
        best_novelty = max(all_results, key=lambda x: x["novelty_rate"])
        best_diversity = max(all_results, key=lambda x: x["diversity"])

        f.write(
            f"1. {best_novelty['model_name']} achieved the highest novelty rate "
            f"({best_novelty['novelty_rate']:.4f}),\n"
        )
        f.write(f"   meaning it generates the most names not seen during training.\n")
        f.write(
            f"   This suggests better generalization beyond memorized patterns.\n\n"
        )

        f.write(
            f"2. {best_diversity['model_name']} achieved the highest diversity "
            f"({best_diversity['diversity']:.4f}),\n"
        )
        f.write(
            f"   producing the most unique names across {NUM_GENERATE} generations.\n"
        )
        f.write(
            f"   This indicates greater variety in the learned character distributions.\n\n"
        )

        # Check if BLSTM overfits (very low training loss can mean memorization)
        blstm_result = next(
            (r for r in all_results if r["model_name"] == "BLSTM"), None
        )
        if blstm_result:
            if blstm_result["novelty_rate"] < 0.5:
                f.write(
                    "3. The BLSTM shows lower novelty, which may indicate overfitting.\n"
                )
                f.write(
                    "   Its bidirectional architecture with many parameters can memorize\n"
                )
                f.write(
                    "   training data more easily, leading to reproductions rather than\n"
                )
                f.write("   novel generations.\n\n")
            else:
                f.write(
                    "3. The BLSTM's bidirectional architecture captures both forward and\n"
                )
                f.write(
                    "   backward context, which helps in learning valid character patterns.\n\n"
                )

        attn_result = next(
            (r for r in all_results if r["model_name"] == "RNN + Attention"), None
        )
        if attn_result:
            f.write("4. The attention mechanism allows the RNN to focus on relevant\n")
            f.write("   positions in the sequence when predicting each character,\n")
            f.write(
                "   potentially improving the quality and variety of generated names.\n\n"
            )

    print(f"\nResults saved to: {results_path}")

    # Step 8: Print Observations
    print("\n" + "=" * 65)
    print("OBSERVATIONS")
    print("=" * 65)

    best_novelty = max(all_results, key=lambda x: x["novelty_rate"])
    best_diversity = max(all_results, key=lambda x: x["diversity"])

    print(f"\nObservation 1:")
    print(
        f"  {best_novelty['model_name']} achieved the highest novelty rate "
        f"({best_novelty['novelty_rate']:.4f}),"
    )
    print(f"  generating the most names not present in the training dataset.")

    print(f"\nObservation 2:")
    print(
        f"  {best_diversity['model_name']} achieved the highest diversity "
        f"({best_diversity['diversity']:.4f}),"
    )
    print(f"  producing the most unique names, indicating greater generation variety.")

    blstm_result = next((r for r in all_results if r["model_name"] == "BLSTM"), None)
    if blstm_result and blstm_result["novelty_rate"] < 0.5:
        print(f"\nObservation 3:")
        print(
            f"  The BLSTM shows lower novelty ({blstm_result['novelty_rate']:.4f}), "
            f"which may indicate overfitting."
        )
        print(f"  Its large parameter count (bidirectional) can lead to memorization")
        print(f"  of training names rather than creative generation.")
    elif blstm_result:
        print(f"\nObservation 3:")
        print(f"  The BLSTM's bidirectional context capture helps learn valid")
        print(f"  character patterns from both directions of the name.")

    print(f"\nObservation 4:")
    print(f"  The attention-based model can focus on relevant character positions,")
    print(f"  improving its ability to model long-range dependencies in names.")

    print("\n" + "=" * 65)
    print("Task-2 evaluation complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
