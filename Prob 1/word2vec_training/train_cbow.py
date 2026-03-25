import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ensure project root is on the path
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from utils.vocabulary import load_corpus, build_vocabulary
from utils.dataset import generate_cbow_pairs, CBOWDataset
from models.cbow import CBOW


def train_cbow(
    corpus_path,
    embedding_dim=100,
    window_size=4,
    min_count=2,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    save_dir=None,
    verbose=True,
):
    # 1. Load corpus and build vocabulary
    sentences = load_corpus(corpus_path)
    word_to_index, index_to_word, word_freq = build_vocabulary(
        sentences, min_count=min_count
    )
    vocab_size = len(word_to_index)

    if verbose:
        print(f"[CBOW] Vocabulary size : {vocab_size}")
        print(f"[CBOW] Embedding dim   : {embedding_dim}")
        print(f"[CBOW] Window size     : {window_size}")

    # 2. Generate training pairs
    pairs = generate_cbow_pairs(sentences, word_to_index, window_size)
    if verbose:
        print(f"[CBOW] Training pairs  : {len(pairs):,}")

    # wrap in a Dataset / DataLoader for batching
    max_context_len = 2 * window_size
    dataset = CBOWDataset(pairs, max_context_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialise model, loss, optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBOW(vocab_size, embedding_dim).to(device)

    # CrossEntropyLoss combines log-softmax + NLL in one fused op,
    # which is numerically more stable than doing them separately.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training loop─
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for ctx, mask, tgt in loader:
            ctx = ctx.to(device)
            mask = mask.to(device)
            tgt = tgt.to(device)

            # forward pass: predict center word from context
            logits = model(ctx, mask)
            loss = criterion(logits, tgt)

            # backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        loss_history.append(avg_loss)

        if verbose:
            print(
                f"  Epoch {epoch:>2d}/{epochs} - "
                f"Loss: {avg_loss:.4f}  ({elapsed:.1f}s)"
            )

    # 5. Save model and embeddings
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "trained_cbow_model.pt")
        torch.save(model.state_dict(), model_path)
        if verbose:
            print(f"[CBOW] Model saved to {model_path}")

    # Build the result dict
    result = {
        "model": model,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "window_size": window_size,
    }
    return result


# Standalone entry point — quick default training run
if __name__ == "__main__":
    corpus = os.path.join(PROJ_ROOT, "data", "clean_corpus.txt")
    out_dir = os.path.join(PROJ_ROOT, "data")

    result = train_cbow(
        corpus_path=corpus,
        embedding_dim=100,
        window_size=4,
        epochs=10,
        batch_size=64,
        save_dir=out_dir,
    )
    print(f"\n[CBOW] Final loss: {result['final_loss']:.4f}")
