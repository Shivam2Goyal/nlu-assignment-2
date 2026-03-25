import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.dataset import load_names

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "TrainingNames.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "qualitative_analysis.txt")

# Common Indian name endings used to judge phonetic realism
COMMON_ENDINGS = [
    "a",
    "ya",
    "an",
    "ar",
    "ish",
    "it",
    "av",
    "in",
    "hi",
    "ka",
    "na",
    "sh",
    "vi",
    "al",
    "ni",
    "ti",
    "ha",
    "ri",
    "am",
    "uk",
    "ur",
    "aj",
    "ik",
    "ee",
]


def load_generated(filename):
    """Load generated names from a file, one per line."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def has_repeated_chars(name, threshold=3):
    """Check if a name has the same character repeated >= threshold times consecutively."""
    for i in range(len(name) - threshold + 1):
        if len(set(name[i : i + threshold])) == 1:
            return True
    return False


def has_unnatural_combos(name):
    """Check for consonant clusters that are rare/absent in Indian names."""
    # Lowercased bigrams very unlikely in Indian names
    unnatural = {
        "xr",
        "qr",
        "zx",
        "xk",
        "qk",
        "bz",
        "fz",
        "kq",
        "xq",
        "zb",
        "zg",
        "zk",
        "ww",
        "bv",
        "gv",
        "fk",
        "bk",
        "wk",
        "vb",
        "fd",
        "wr",
        "bw",
        "gw",
        "dg",
        "kg",
        "fv",
        "kw",
        "vk",
        "bh" not in {"bh"},  # bh is valid in Indian names
        "fh",
        "gh" not in {"gh"},
    }
    # Simpler approach: check for unlikely consonant-only trigrams
    vowels = set("aeiou")
    low = name.lower()
    consonant_run = 0
    for ch in low:
        if ch in vowels:
            consonant_run = 0
        else:
            consonant_run += 1
        if consonant_run >= 4:
            return True
    return False


def is_truncated(name):
    """Check if a name is very short (1-2 chars), suggesting premature <END>."""
    return len(name) <= 2


def is_overly_long(name, threshold=12):
    """Check if a name is unusually long for an Indian first name."""
    return len(name) > threshold


def ends_with_common_suffix(name):
    """Check if the name ends with a common Indian name suffix."""
    low = name.lower()
    return any(low.endswith(ending) for ending in COMMON_ENDINGS)


def analyze_names(names, model_name, training_set):
    """
    Perform full qualitative analysis on a list of generated names.
    Returns a dict of analysis results.
    """
    total = len(names)

    # Failure modes
    repeated = [n for n in names if has_repeated_chars(n)]
    unnatural = [n for n in names if has_unnatural_combos(n)]
    truncated = [n for n in names if is_truncated(n)]
    too_long = [n for n in names if is_overly_long(n)]
    memorized = [n for n in names if n.strip().lower() in training_set]

    # Realism indicators
    valid_endings = [n for n in names if ends_with_common_suffix(n)]
    avg_len = sum(len(n) for n in names) / total if total else 0
    len_distribution = {}
    for n in names:
        bucket = len(n)
        len_distribution[bucket] = len_distribution.get(bucket, 0) + 1

    return {
        "model_name": model_name,
        "total": total,
        "repeated_chars": repeated,
        "unnatural_combos": unnatural,
        "truncated": truncated,
        "too_long": too_long,
        "memorized": memorized,
        "valid_endings": valid_endings,
        "avg_length": avg_len,
        "len_distribution": dict(sorted(len_distribution.items())),
    }


def print_and_write(text, f):
    """Print to console and write to file simultaneously."""
    print(text)
    f.write(text + "\n")


def main():
    random.seed(42)

    # Load training set for comparison
    training_names = load_names(DATA_PATH)
    training_set = set(n.strip().lower() for n in training_names)
    training_avg_len = sum(len(n) for n in training_names) / len(training_names)

    # Load generated names from Task-2
    model_files = {
        "Vanilla RNN": "rnn_generated.txt",
        "BLSTM": "blstm_generated.txt",
        "RNN + Attention": "attention_generated.txt",
    }

    all_names = {}
    all_analysis = {}
    for model_name, filename in model_files.items():
        names = load_generated(filename)
        all_names[model_name] = names
        all_analysis[model_name] = analyze_names(names, model_name, training_set)

    # Output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        def p(text=""):
            print_and_write(text, f)

        p("=" * 70)
        p("Task-3: Qualitative Analysis of Generated Indian Names")
        p("=" * 70)

        #  Step 1: Representative Samples
        p("\n" + "=" * 70)
        p("STEP 1 — REPRESENTATIVE SAMPLES (25 per model)")
        p("=" * 70)

        for model_name, names in all_names.items():
            p(f"\n{model_name} Samples")
            p("-" * 40)
            sampled = random.sample(names, min(25, len(names)))
            for i, name in enumerate(sampled, 1):
                p(f"  {i:2d}. {name}")

        #  Step 2: Realism Analysis
        p("\n" + "=" * 70)
        p("STEP 2 — REALISM ANALYSIS")
        p("=" * 70)

        p(f"\nTraining set reference:")
        p(f"  Average name length: {training_avg_len:.1f} characters")

        for model_name, a in all_analysis.items():
            valid_pct = len(a["valid_endings"]) / a["total"] * 100
            p(f"\n{model_name}:")
            p(f"  Average generated name length : {a['avg_length']:.1f} characters")
            p(
                f"  Names with common Indian endings: {len(a['valid_endings'])}/{a['total']} "
                f"({valid_pct:.1f}%)"
            )
            p(f"  Length distribution (length: count):")
            for length, count in a["len_distribution"].items():
                bar = "#" * min(count // 5, 40)
                p(f"    {length:2d}: {count:4d} {bar}")

        # Observations on realism
        p("\nRealism Observations:")
        p("-" * 40)

        rnn_a = all_analysis["Vanilla RNN"]
        blstm_a = all_analysis["BLSTM"]
        attn_a = all_analysis["RNN + Attention"]

        # Collect realistic-looking names from RNN (have valid endings, len 4-8)
        rnn_realistic = [
            n
            for n in all_names["Vanilla RNN"]
            if ends_with_common_suffix(n)
            and 4 <= len(n) <= 9
            and not has_unnatural_combos(n)
        ][:8]
        attn_realistic = [
            n
            for n in all_names["RNN + Attention"]
            if ends_with_common_suffix(n)
            and 4 <= len(n) <= 9
            and not has_unnatural_combos(n)
        ][:8]
        blstm_realistic = [
            n
            for n in all_names["BLSTM"]
            if ends_with_common_suffix(n)
            and 4 <= len(n) <= 9
            and not has_unnatural_combos(n)
        ][:8]

        p(f"\n  Vanilla RNN produces many names that follow valid Indian phonetic")
        p(f"  patterns and common name endings (-a, -ya, -an, -ish, -ka, -ti).")
        p(f"  Realistic examples: {', '.join(rnn_realistic)}")

        p(f"\n  BLSTM generates names that are mostly nonsensical. Because the model")
        p(f"  is trained bidirectionally but generates left-to-right using only the")
        p(f"  forward direction, outputs lack coherent Indian phonetics.")
        if blstm_realistic:
            p(f"  Few passable examples: {', '.join(blstm_realistic)}")
        else:
            p(f"  Very few names resemble real Indian names.")

        p(f"\n  RNN + Attention generates the most natural-sounding names overall.")
        p(f"  The attention mechanism helps capture valid character dependencies,")
        p(f"  producing names with proper Indian name structure.")
        p(f"  Realistic examples: {', '.join(attn_realistic)}")

        #  Step 3: Failure Modes
        p("\n" + "=" * 70)
        p("STEP 3 — FAILURE MODES")
        p("=" * 70)

        for model_name, a in all_analysis.items():
            p(f"\n{model_name}")
            p("-" * 40)

            # 1. Repeated characters
            p(f"  1. Repeated characters: {len(a['repeated_chars'])} names")
            if a["repeated_chars"]:
                examples = a["repeated_chars"][:5]
                p(f"     Examples: {', '.join(examples)}")

            # 2. Unnatural character combinations
            p(f"  2. Unnatural consonant clusters: {len(a['unnatural_combos'])} names")
            if a["unnatural_combos"]:
                examples = a["unnatural_combos"][:5]
                p(f"     Examples: {', '.join(examples)}")

            # 3. Truncated names
            p(f"  3. Truncated names (<=2 chars): {len(a['truncated'])} names")
            if a["truncated"]:
                p(f"     Examples: {', '.join(a['truncated'][:10])}")

            # 4. Overly long names
            p(f"  4. Overly long names (>12 chars): {len(a['too_long'])} names")
            if a["too_long"]:
                examples = a["too_long"][:5]
                p(f"     Examples: {', '.join(examples)}")

            # 5. Memorized training names
            p(f"  5. Memorized training names: {len(a['memorized'])} names")
            if a["memorized"]:
                examples = list(set(a["memorized"]))[:10]
                p(f"     Examples: {', '.join(examples)}")

        # Failure mode summary table
        p(f"\nFailure Mode Summary Table:")
        p(
            f"  {'Model':<20} {'Repeated':>10} {'Unnatural':>10} {'Truncated':>10} "
            f"{'Too Long':>10} {'Memorized':>10}"
        )
        p(f"  {'-'*72}")
        for model_name, a in all_analysis.items():
            p(
                f"  {model_name:<20} {len(a['repeated_chars']):>10} "
                f"{len(a['unnatural_combos']):>10} {len(a['truncated']):>10} "
                f"{len(a['too_long']):>10} {len(a['memorized']):>10}"
            )

        #  Step 4: Model Comparison
        p("\n" + "=" * 70)
        p("STEP 4 — MODEL COMPARISON")
        p("=" * 70)

        p("\n  Vanilla RNN:")
        p("    Produces recognizable Indian-sounding names with valid phonetic")
        p("    patterns. Occasionally generates repeated characters or slightly")
        p("    unnatural combinations. Some training names are memorized (~9%),")
        p("    showing the model has learned real patterns but also reproduces")
        p("    known names. Average name length is close to training data.")

        p("\n  BLSTM:")
        p("    Generates highly novel but largely unrealistic names. The mismatch")
        p("    between bidirectional training and left-to-right generation causes")
        p("    outputs to contain many consonant clusters and character patterns")
        p("    absent in Indian names. While novelty and diversity scores are high")
        p("    (1.0 and 0.985), this reflects poor generation quality rather than")
        p("    creative generalization. Zero training names are memorized because")
        p("    the generation process diverges from learned distributions.")

        p("\n  RNN + Attention:")
        p("    Produces the most realistic and consistent Indian-sounding names.")
        p("    The attention mechanism allows the model to focus on relevant")
        p("    previously-seen characters when predicting the next one, resulting")
        p("    in coherent phonetic sequences. It has slightly more memorized")
        p("    training names (~10.5%), indicating strong pattern learning.")
        p("    Names with common Indian suffixes (-a, -ka, -ni, -sh) appear")
        p("    frequently, suggesting effective capture of naming conventions.")

        #  Step 5: Quantitative + Qualitative
        p("\n" + "=" * 70)
        p("STEP 5 — COMBINING QUANTITATIVE AND QUALITATIVE INSIGHTS")
        p("=" * 70)

        p(
            """
  Quantitative metrics from Task-2:

    Model              Novelty Rate    Diversity
    ------------------------------------------------
    Vanilla RNN           0.9100         0.9310
    BLSTM                 1.0000         0.9850
    RNN + Attention       0.8949         0.9139

  The BLSTM achieved the highest novelty (1.0) and diversity (0.985),
  but qualitative analysis reveals this is misleading — its generated
  names are mostly nonsensical (e.g., "Cozoav", "Uobgar", "Iosubv").
  High novelty here signals poor generation fidelity, not creativity.

  The Vanilla RNN and RNN + Attention both produce realistic names
  with moderate novelty (~0.89–0.91) and diversity (~0.91–0.93).
  Their lower novelty reflects the fact that some generated names
  correctly match real training names — a sign of good learning.

  The RNN + Attention model strikes the best balance: it generates
  diverse, realistic names while maintaining strong novelty. Its
  attention mechanism provides better context modeling than the
  vanilla RNN, leading to more coherent character sequences."""
        )

        #  Step 6: Conclusion
        p("\n" + "=" * 70)
        p("STEP 6 — CONCLUSION")
        p("=" * 70)

        p(
            """
  1. The RNN + Attention model produces the most realistic Indian names,
     with proper phonetic patterns and common name suffixes. The attention
     mechanism effectively captures character-level dependencies, leading
     to coherent and natural-sounding outputs like "Monita", "Laksha",
     "Sulita", and "Adrini".

  2. The Vanilla RNN generates reasonably good names but occasionally
     produces truncated or slightly awkward character combinations. Its
     simpler architecture limits its ability to model long-range
     dependencies within names.

  3. The BLSTM model, despite having the most parameters (179,314) and
     achieving the lowest training loss (0.0086), performs worst in
     qualitative terms. The fundamental issue is the train-generate
     mismatch: bidirectional training captures context from both
     directions, but autoregressive generation only uses the forward
     direction. This architectural limitation makes BLSTM unsuitable
     for open-ended generation tasks without additional design changes
     (e.g., using a decoder or scheduled sampling).

  4. Quantitative metrics alone are insufficient for evaluating
     generation quality. The BLSTM's perfect novelty score (1.0)
     is a cautionary example — high novelty can indicate failure
     rather than success when generated outputs lack realism.

  5. For character-level Indian name generation, the RNN + Attention
     architecture offers the best trade-off between quality, diversity,
     and novelty, making it the recommended approach among the three."""
        )

        p("\n" + "=" * 70)
        p("End of Qualitative Analysis — Problem 2 Complete")
        p("=" * 70)

    print(f"\nAnalysis saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
