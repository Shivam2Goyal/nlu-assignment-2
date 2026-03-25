import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Semantic groups for color-coding 
# Each group maps a human-readable label to the words that belong
# to it.  Words not in any group are drawn in grey.

SEMANTIC_GROUPS = {
    "Degree programs": [
        "btech",
        "mtech",
        "phd",
        "mba",
        "ug",
        "pg",
    ],
    "Academic / student life": [
        "student",
        "faculty",
        "exam",
        "course",
        "program",
        "department",
        "semester",
        "academic",
        "curriculum",
        "professor",
        "dean",
        "director",
    ],
    "Research & innovation": [
        "research",
        "laboratory",
        "project",
        "innovation",
        "technology",
        "development",
        "conference",
        "publication",
        "journal",
        "workshop",
    ],
    "Campus & institution": [
        "iit",
        "jodhpur",
        "campus",
        "institute",
        "education",
        "library",
        "sports",
        "cultural",
        "committee",
        "council",
    ],
    "Career & training": [
        "engineering",
        "science",
        "knowledge",
        "training",
        "scholarship",
        "placement",
    ],
}

# Color palette (one color per group + grey for uncategorized)
GROUP_COLORS = {
    "Degree programs": "#e41a1c",  # red
    "Academic / student life": "#377eb8",  # blue
    "Research & innovation": "#4daf4a",  # green
    "Campus & institution": "#ff7f00",  # orange
    "Career & training": "#984ea3",  # purple
}
DEFAULT_COLOR = "#999999"  # grey for words not in any group


def _word_to_group(word):
    """Return the semantic group label for a word, or None."""
    for group, members in SEMANTIC_GROUPS.items():
        if word in members:
            return group
    return None


def plot_embeddings(reduced_vectors, words, title, save_path):
    # Create a labeled scatter plot of 2-D word vectors, color-coded by semantic group, and save to disk.
    fig, ax = plt.subplots(figsize=(14, 10))

    # Track which groups we've added to the legend
    legend_entries = {}

    for i, word in enumerate(words):
        x, y = reduced_vectors[i]
        group = _word_to_group(word)
        color = GROUP_COLORS.get(group, DEFAULT_COLOR)
        label = group if group and group not in legend_entries else None

        ax.scatter(
            x, y, c=color, s=60, edgecolors="k", linewidths=0.3, label=label, zorder=3
        )
        # Place the word label slightly offset so it doesn't overlap the dot
        ax.annotate(
            word,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            ha="left",
            va="bottom",
        )

        if group and group not in legend_entries:
            legend_entries[group] = True

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"  Saved: {save_path}")
