"""
statistics.py — Corpus Statistics and Frequency Analysis
=========================================================
After the corpus has been cleaned and tokenized, this module
computes the dataset-level statistics that the assignment asks for:

    1. Total number of documents
    2. Total number of tokens (across all documents)
    3. Vocabulary size (unique token count)
    4. Top-20 most frequent words

All results are printed to the terminal in a readable format.
"""

from collections import Counter


def compute_statistics(corpus_tokens):
    """
    Walk through the tokenized corpus and compute basic counts.

    Parameters
    ----------
    corpus_tokens : list[list[str]]
        Each inner list holds the tokens of one document.

    Returns
    -------
    stats : dict
        Dictionary with keys:
            'num_docs'      — number of documents
            'num_tokens'    — total token count
            'vocab_size'    — unique token count
            'word_freq'     — Counter object with per-word counts
    """
    num_docs = len(corpus_tokens)

    # flatten all tokens into a single list to count them
    all_tokens = []
    for doc in corpus_tokens:
        all_tokens.extend(doc)

    num_tokens = len(all_tokens)

    # count frequency of each unique word
    word_freq = Counter(all_tokens)
    vocab_size = len(word_freq)

    stats = {
        "num_docs": num_docs,
        "num_tokens": num_tokens,
        "vocab_size": vocab_size,
        "word_freq": word_freq,
    }
    return stats


def print_statistics(stats):
    """
    Pretty-print the dataset statistics to the console.
    The format matches the assignment's expected output.
    """
    print("\n" + "=" * 45)
    print("       Dataset Statistics")
    print("=" * 45)
    print(f"  Total Documents  : {stats['num_docs']}")
    print(f"  Total Tokens     : {stats['num_tokens']:,}")
    print(f"  Vocabulary Size  : {stats['vocab_size']:,}")
    print("=" * 45)

    # show the 20 most common words
    print("\n  Top 20 Most Frequent Words")
    print("  " + "-" * 35)
    for rank, (word, count) in enumerate(stats["word_freq"].most_common(20), start=1):
        print(f"  {rank:>3d}. {word:20s} : {count}")
    print()
