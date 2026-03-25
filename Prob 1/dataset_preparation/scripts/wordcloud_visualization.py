"""
wordcloud_visualization.py — Word Cloud Generator
===================================================
Generates a word-cloud image from the corpus token frequencies.

For the visualisation we DO remove generic stopwords (the, is,
and, of, …) so that the cloud highlights domain-specific terms
like "research", "students", "iit", "jodhpur", etc.

The resulting image is saved as wordcloud.png.
"""

import os
from wordcloud import WordCloud
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt


# A hand-picked set of English stopwords to suppress in the cloud.
# We keep this list self-contained so the script doesn't depend on
# NLTK data downloads at runtime.
STOPWORDS_FOR_CLOUD = {
    "the",
    "is",
    "in",
    "it",
    "and",
    "or",
    "of",
    "to",
    "a",
    "an",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "can",
    "could",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "we",
    "they",
    "its",
    "his",
    "her",
    "our",
    "their",
    "them",
    "not",
    "no",
    "but",
    "so",
    "if",
    "than",
    "then",
    "also",
    "more",
    "all",
    "any",
    "each",
    "every",
    "such",
    "about",
    "up",
    "out",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "how",
    "there",
    "here",
    "into",
    "through",
    "during",
    "after",
    "before",
    "between",
    "under",
    "over",
    "above",
    "below",
    "other",
    "some",
    "only",
    "very",
}


def generate_wordcloud(word_freq, output_path):
    """
    Build and save a word-cloud image.

    Parameters
    ----------
    word_freq : collections.Counter
        Token frequencies from the corpus (produced by statistics.py).
    output_path : str
        File path where the PNG image will be written.
    """
    # filter out stopwords from the frequency dict so they don't
    # dominate the visual
    filtered_freq = {
        word: count
        for word, count in word_freq.items()
        if word not in STOPWORDS_FOR_CLOUD
    }

    # safety check — need at least a few words to draw a cloud
    if len(filtered_freq) < 5:
        print("[warn] Not enough words to generate a meaningful cloud.")
        return

    # create the word cloud object
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",  # a pleasant colour palette
        prefer_horizontal=0.7,
        min_font_size=8,
    )
    wc.generate_from_frequencies(filtered_freq)

    # plot and save
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("IIT Jodhpur Corpus — Word Cloud", fontsize=16, pad=12)

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Word cloud saved to {output_path}")
