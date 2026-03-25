import os
import sys

# make sure the scripts/ package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.scraper import scrape_all_sources
from scripts.preprocess import (
    build_corpus,
    save_clean_corpus,
    save_corpus_pickle,
)
from scripts.statistics import compute_statistics, print_statistics
from scripts.wordcloud_visualization import generate_wordcloud

# output paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_pages")
CORPUS_TXT = os.path.join(DATA_DIR, "clean_corpus.txt")
CORPUS_PKL = os.path.join(DATA_DIR, "corpus_tokens.pkl")
WORDCLOUD_IMG = os.path.join(SCRIPT_DIR, "wordcloud.png")


def main():
    """
    Runs every stage of the pipeline in sequence and prints
    progress messages so you can watch it work.
    """

    # Stage 1: Scrape
    print("=" * 55)
    print("  STAGE 1 — Scraping IIT Jodhpur web pages")
    print("=" * 55)
    documents = scrape_all_sources(output_dir=RAW_DIR)

    if len(documents) == 0:
        print(
            "\n[FATAL] No pages were collected. Check your internet "
            "connection and try again."
        )
        sys.exit(1)

    # Stage 2: Preprocess
    print("\n" + "=" * 55)
    print("  STAGE 2 — Cleaning and tokenizing")
    print("=" * 55)
    corpus_tokens = build_corpus(documents)

    # Stage 3: Save corpus files
    print("\n" + "=" * 55)
    print("  STAGE 3 — Saving corpus files")
    print("=" * 55)
    save_clean_corpus(corpus_tokens, CORPUS_TXT)
    save_corpus_pickle(corpus_tokens, CORPUS_PKL)

    # Stage 4: Statistics
    print("\n" + "=" * 55)
    print("  STAGE 4 — Computing dataset statistics")
    print("=" * 55)
    stats = compute_statistics(corpus_tokens)
    print_statistics(stats)

    # Stage 5: Word cloud
    print("=" * 55)
    print("  STAGE 5 — Generating word cloud")
    print("=" * 55)
    generate_wordcloud(stats["word_freq"], WORDCLOUD_IMG)

    print("\n✓ Pipeline finished. Output files:")
    print(f"    {CORPUS_TXT}")
    print(f"    {CORPUS_PKL}")
    print(f"    {WORDCLOUD_IMG}")


if __name__ == "__main__":
    main()
