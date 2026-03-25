import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# ensure project root is on the path
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from utils.vocabulary import (
    load_corpus,
    build_vocabulary,
    compute_unigram_distribution,
)
from utils.dataset import generate_skipgram_pairs, SkipGramDataset
from utils.negative_sampling import NegativeSampler
from models.skipgram import SkipGram


def train_skipgram(
    corpus_path,
    embedding_dim=100,
    window_size=4,
    num_negative_samples=5,
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
        print(f"[SGNS] Vocabulary size    : {vocab_size}")
        print(f"[SGNS] Embedding dim      : {embedding_dim}")
        print(f"[SGNS] Window size        : {window_size}")
        print(f"[SGNS] Negative samples   : {num_negative_samples}")

    # 2. Build unigram distribution and negative sampler
    unigram_probs = compute_unigram_distribution(word_freq, word_to_index, power=0.75)
    sampler = NegativeSampler(unigram_probs, vocab_size)

    # 3. Generate training pairs
    pairs = generate_skipgram_pairs(sentences, word_to_index, window_size)
    if verbose:
        print(f"[SGNS] Training pairs     : {len(pairs):,}")

    dataset = SkipGramDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Initialise model and optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkipGram(vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training loop
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for center, context in loader:
            center = center.to(device)
            context = context.to(device)

            # draw negative samples for this batch
            neg_ids = sampler.sample_batch(
                center.size(0), num_negative_samples, context
            ).to(device)

            # forward pass: the model computes the NS loss internally
            loss = model(center, context, neg_ids)

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

    # 6. Save model
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "trained_skipgram_model.pt")
        torch.save(model.state_dict(), model_path)
        if verbose:
            print(f"[SGNS] Model saved to {model_path}")

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
        "num_negative_samples": num_negative_samples,
    }
    return result


# Standalone entry point — quick default training run
if __name__ == "__main__":
    corpus = os.path.join(PROJ_ROOT, "data", "clean_corpus.txt")
    out_dir = os.path.join(PROJ_ROOT, "data")

    result = train_skipgram(
        corpus_path=corpus,
        embedding_dim=100,
        window_size=4,
        num_negative_samples=5,
        epochs=10,
        batch_size=64,
        save_dir=out_dir,
    )
    print(f"\n[SGNS] Final loss: {result['final_loss']:.4f}")
