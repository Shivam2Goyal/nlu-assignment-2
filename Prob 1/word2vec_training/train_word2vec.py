import os
import re
import csv
import json
import random
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_corpus(corpus_path: str):
    text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore").lower()

    chunks = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        # Split long blocks further on punctuation.
        parts = re.split(r"(?<=[.!?])\s+", block)
        for part in parts:
            part = part.strip()
            if part:
                chunks.append(part)

    return chunks if chunks else [text]


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def build_sentences(corpus_path: str):
    raw_chunks = read_corpus(corpus_path)
    sentences = [tokenize(chunk) for chunk in raw_chunks]
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


class Vocabulary:
    def __init__(self, sentences, min_count=2):
        counter = Counter(tok for sent in sentences for tok in sent)

        # 0 = padding, 1 = unknown
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": 0, "<unk>": 1}

        for word, freq in counter.most_common():
            if freq >= min_count:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

        self.counts = Counter({w: c for w, c in counter.items() if w in self.stoi})
        self.total_tokens = sum(counter.values())

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]  # 1 = <unk>

    def decode(self, indices):
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in indices]


class CBOWDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=5):
        self.samples = []
        for sent in sentences:
            ids = vocab.encode(sent)
            n = len(ids)
            for i, target in enumerate(ids):
                left = max(0, i - window_size)
                right = min(n, i + window_size + 1)
                context = ids[left:i] + ids[i + 1:right]
                if context:
                    self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SkipGramDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=5):
        self.samples = []
        for sent in sentences:
            ids = vocab.encode(sent)
            n = len(ids)
            for i, center in enumerate(ids):
                left = max(0, i - window_size)
                right = min(n, i + window_size + 1)
                for j in range(left, right):
                    if j != i:
                        self.samples.append((center, ids[j]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


def cbow_collate(batch):
    contexts, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in contexts], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ctx in enumerate(contexts):
        padded[i, :len(ctx)] = ctx
    targets = torch.stack(targets)
    return padded, lengths, targets


def skipgram_collate(batch):
    centers, contexts = zip(*batch)
    return torch.stack(centers), torch.stack(contexts)


class NegativeSampler:
    def __init__(self, vocab, power=0.75):
        freqs = np.array([vocab.counts.get(tok, 1) for tok in vocab.itos], dtype=np.float64)
        freqs[0] = 0.0  # never sample padding
        probs = freqs ** power
        probs = probs / probs.sum()
        self.probs = torch.tensor(probs, dtype=torch.float32)

    def sample(self, batch_size, num_negatives, exclude=None, device="cpu"):
        neg = torch.multinomial(self.probs, batch_size * num_negatives, replacement=True)
        neg = neg.view(batch_size, num_negatives)

        # Avoid sampling the positive word as a negative sample.
        if exclude is not None:
            exclude = exclude.view(-1, 1)
            mask = neg.eq(exclude)
            while mask.any():
                resample = torch.multinomial(self.probs, int(mask.sum().item()), replacement=True)
                neg[mask] = resample
                mask = neg.eq(exclude)

        return neg.to(device)


class Word2VecNS(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.5 / self.input_emb.embedding_dim
        nn.init.uniform_(self.input_emb.weight, -init_range, init_range)
        nn.init.zeros_(self.output_emb.weight)
        self.input_emb.weight.data[0].zero_()  # keep <pad> fixed

    def loss_from_vectors(self, center_vecs, pos_idx, neg_idx):
        pos_vecs = self.output_emb(pos_idx)
        pos_score = torch.sum(center_vecs * pos_vecs, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_vecs = self.output_emb(neg_idx)  # [B, K, D]
        neg_score = torch.bmm(neg_vecs, center_vecs.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss)
        return loss.mean()

    def get_embeddings(self):
        return self.input_emb.weight.detach().cpu().numpy()


@torch.no_grad()
def evaluate_loss(model, loader, sampler, device, mode, num_negatives):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        if mode == "cbow":
            contexts, lengths, targets = batch
            contexts = contexts.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            emb = model.input_emb(contexts)
            mask = (contexts != 0).unsqueeze(-1).float()
            summed = (emb * mask).sum(dim=1)
            lengths = lengths.clamp(min=1).unsqueeze(1).float()
            center_vecs = summed / lengths

            neg = sampler.sample(targets.size(0), num_negatives, exclude=targets, device=device)
            loss = model.loss_from_vectors(center_vecs, targets, neg)

        else:
            centers, contexts = batch
            centers = centers.to(device)
            contexts = contexts.to(device)

            center_vecs = model.input_emb(centers)
            neg = sampler.sample(centers.size(0), num_negatives, exclude=contexts, device=device)
            loss = model.loss_from_vectors(center_vecs, contexts, neg)

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def train_model(sentences, vocab, model_name, window_size=5, emb_dim=100, num_negatives=5,
                epochs=10, batch_size=256, lr=0.003, device="cpu"):
    sampler = NegativeSampler(vocab)

    if model_name == "cbow":
        dataset = CBOWDataset(sentences, vocab, window_size=window_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=cbow_collate)
    elif model_name == "skipgram":
        dataset = SkipGramDataset(sentences, vocab, window_size=window_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=skipgram_collate)
    else:
        raise ValueError("model_name must be 'cbow' or 'skipgram'")

    model = Word2VecNS(len(vocab), emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for batch in loader:
            optimizer.zero_grad()

            if model_name == "cbow":
                contexts, lengths, targets = batch
                contexts = contexts.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)

                emb = model.input_emb(contexts)
                mask = (contexts != 0).unsqueeze(-1).float()
                summed = (emb * mask).sum(dim=1)
                lengths = lengths.clamp(min=1).unsqueeze(1).float()
                center_vecs = summed / lengths

                neg = sampler.sample(targets.size(0), num_negatives, exclude=targets, device=device)
                loss = model.loss_from_vectors(center_vecs, targets, neg)

            else:
                centers, contexts = batch
                centers = centers.to(device)
                contexts = contexts.to(device)

                center_vecs = model.input_emb(centers)
                neg = sampler.sample(centers.size(0), num_negatives, exclude=contexts, device=device)
                loss = model.loss_from_vectors(center_vecs, contexts, neg)

            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1

        avg_train = running / max(1, steps)
        avg_eval = evaluate_loss(model, loader, sampler, device, model_name, num_negatives)
        history.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": avg_train,
            "eval_loss": avg_eval
        })
        print(f"[{model_name.upper()}] epoch {epoch:02d}/{epochs} | train_loss={avg_train:.4f} | eval_loss={avg_eval:.4f}")

    return model, history


def save_embeddings(model, vocab, out_path):
    data = {
        "stoi": vocab.stoi,
        "itos": vocab.itos,
        "embeddings": model.get_embeddings(),
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def write_history_csv(rows, out_csv):
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to clean_corpus.txt")
    parser.add_argument("--out_dir", required=True, help="Directory to store models/results")
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = build_sentences(args.corpus)
    vocab = Vocabulary(sentences, min_count=args.min_count)

    stats = {
        "documents": len(sentences),
        "total_tokens": sum(len(s) for s in sentences),
        "vocab_size": len(vocab),
        "min_count": args.min_count,
    }
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_history = []

    for model_name in ["cbow", "skipgram"]:
        model, history = train_model(
            sentences=sentences,
            vocab=vocab,
            model_name=model_name,
            window_size=args.window_size,
            emb_dim=args.emb_dim,
            num_negatives=args.num_negatives,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        torch.save(model.state_dict(), out_dir / f"trained_{model_name}_model.pt")
        save_embeddings(model, vocab, out_dir / f"{model_name}_embeddings.pkl")
        all_history.extend(history)

    write_history_csv(all_history, out_dir / "training_results.csv")
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()