import os
import sys
import csv
import pickle
import itertools

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from train_cbow import train_cbow
from train_skipgram import train_skipgram

# Paths
CORPUS_PATH = os.path.join(PROJ_ROOT, "data", "clean_corpus.txt")
OUT_DIR = os.path.join(PROJ_ROOT, "data")

# Hyperparameter grids
EMBEDDING_DIMS = [100, 200, 300]
WINDOW_SIZES = [2, 4, 6]
NEG_SAMPLES = [5, 10]

# Shared training settings
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MIN_COUNT = 2


def run_experiments():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    best_cbow = None
    best_sgns = None

    # CBOW experiments─
    cbow_grid = list(itertools.product(EMBEDDING_DIMS, WINDOW_SIZES))
    total_cbow = len(cbow_grid)

    for run_idx, (emb_dim, win) in enumerate(cbow_grid, 1):
        print(f"\n{'='*60}")
        print(
            f"[Experiment {run_idx}/{total_cbow}]  CBOW  " f"emb={emb_dim}  win={win}"
        )
        print(f"{'='*60}")

        res = train_cbow(
            corpus_path=CORPUS_PATH,
            embedding_dim=emb_dim,
            window_size=win,
            min_count=MIN_COUNT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_dir=None,  # we save only the best model later
            verbose=True,
        )

        row = {
            "model": "CBOW",
            "embedding_dim": emb_dim,
            "window_size": win,
            "negative_samples": "N/A",
            "final_loss": round(res["final_loss"], 4),
        }
        results.append(row)

        if best_cbow is None or res["final_loss"] < best_cbow[0]:
            best_cbow = (res["final_loss"], res)

    # Skip-Gram experiments
    sgns_grid = list(itertools.product(EMBEDDING_DIMS, WINDOW_SIZES, NEG_SAMPLES))
    total_sgns = len(sgns_grid)

    for run_idx, (emb_dim, win, neg_k) in enumerate(sgns_grid, 1):
        print(f"\n{'='*60}")
        print(
            f"[Experiment {run_idx}/{total_sgns}]  SGNS  "
            f"emb={emb_dim}  win={win}  neg={neg_k}"
        )
        print(f"{'='*60}")

        res = train_skipgram(
            corpus_path=CORPUS_PATH,
            embedding_dim=emb_dim,
            window_size=win,
            num_negative_samples=neg_k,
            min_count=MIN_COUNT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_dir=None,
            verbose=True,
        )

        row = {
            "model": "Skip-Gram",
            "embedding_dim": emb_dim,
            "window_size": win,
            "negative_samples": neg_k,
            "final_loss": round(res["final_loss"], 4),
        }
        results.append(row)

        if best_sgns is None or res["final_loss"] < best_sgns[0]:
            best_sgns = (res["final_loss"], res)

    # Save best models
    import torch

    if best_cbow:
        cbow_model = best_cbow[1]["model"]
        torch.save(
            cbow_model.state_dict(),
            os.path.join(OUT_DIR, "trained_cbow_model.pt"),
        )

    if best_sgns:
        sgns_model = best_sgns[1]["model"]
        torch.save(
            sgns_model.state_dict(),
            os.path.join(OUT_DIR, "trained_skipgram_model.pt"),
        )

    # Save word embeddings (from best of each model)
    embeddings_dict = {}

    if best_cbow:
        cbow_emb = best_cbow[1]["model"].get_embeddings().cpu().numpy()
        w2i = best_cbow[1]["word_to_index"]
        i2w = best_cbow[1]["index_to_word"]
        for word, idx in w2i.items():
            embeddings_dict[f"cbow_{word}"] = cbow_emb[idx]

    if best_sgns:
        sgns_emb = best_sgns[1]["model"].get_embeddings().cpu().numpy()
        w2i = best_sgns[1]["word_to_index"]
        for word, idx in w2i.items():
            embeddings_dict[f"sgns_{word}"] = sgns_emb[idx]

    pkl_path = os.path.join(OUT_DIR, "word_embeddings.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    # Save results CSV
    csv_path = os.path.join(OUT_DIR, "training_results.csv")
    fieldnames = [
        "model",
        "embedding_dim",
        "window_size",
        "negative_samples",
        "final_loss",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print(f"\n\n{'='*65}")
    print("  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*65}")
    header = f"{'Model':<12} {'Emb':<5} {'Win':<5} {'Neg':<5} {'Loss':<10}"
    print(header)
    print("-" * 65)
    for r in results:
        print(
            f"{r['model']:<12} {r['embedding_dim']:<5} "
            f"{r['window_size']:<5} {str(r['negative_samples']):<5} "
            f"{r['final_loss']:<10}"
        )
    print(f"{'='*65}")

    if best_cbow:
        bc = best_cbow[1]
        print(
            f"\nBest CBOW      : emb={bc['embedding_dim']}, "
            f"win={bc['window_size']}  ->  loss={best_cbow[0]:.4f}"
        )
    if best_sgns:
        bs = best_sgns[1]
        print(
            f"Best Skip-Gram : emb={bs['embedding_dim']}, "
            f"win={bs['window_size']}, neg={bs['num_negative_samples']}  "
            f"->  loss={best_sgns[0]:.4f}"
        )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {pkl_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'trained_cbow_model.pt')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'trained_skipgram_model.pt')}")

    return results


if __name__ == "__main__":
    run_experiments()
