from collections import Counter


def load_corpus(filepath):
    
    sentences = []
    with open(filepath, "r", encoding="utf-8") as fin:
        for line in fin:
            tokens = line.strip().split()
            if tokens:  # skip blank lines
                sentences.append(tokens)
    return sentences


def build_vocabulary(sentences, min_count=2):
    
    # Step 1 — count every word across all sentences
    raw_counts = Counter()
    for sent in sentences:
        raw_counts.update(sent)

    # Step 2 — keep only words that appear at least min_count times
    # Rare words add noise and will never get enough gradient signal
    # to learn useful embeddings.
    word_frequency = {w: c for w, c in raw_counts.items() if c >= min_count}

    # Step 3 — assign integer indices in a deterministic order
    # Sorting alphabetically ensures reproducible mappings across runs.
    sorted_words = sorted(word_frequency.keys())
    word_to_index = {word: idx for idx, word in enumerate(sorted_words)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    return word_to_index, index_to_word, word_frequency


def compute_unigram_distribution(word_frequency, word_to_index, power=0.75):

    vocab_size = len(word_to_index)
    # initialise with zeros in index order
    raw_powered = [0.0] * vocab_size

    for word, idx in word_to_index.items():
        raw_powered[idx] = word_frequency[word] ** power

    # normalise so the values sum to 1 (a valid probability distribution)
    total = sum(raw_powered)
    unigram_probs = [val / total for val in raw_powered]

    return unigram_probs


# Quick sanity-check when run as a standalone script
if __name__ == "__main__":
    import os

    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "clean_corpus.txt"
    )
    sents = load_corpus(corpus_path)
    w2i, i2w, freq = build_vocabulary(sents, min_count=2)

    print(f"Sentences loaded  : {len(sents)}")
    print(f"Vocabulary size   : {len(w2i)}")
    print(f"Sample mappings   : {list(w2i.items())[:5]}")
    top5 = sorted(freq.items(), key=lambda x: -x[1])[:5]
    print(f"Top-5 words       : {top5}")
