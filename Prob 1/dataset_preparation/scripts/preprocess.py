import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Keep important structural words
IMPORTANT_WORDS = {"not", "no", "nor"}
STOP_WORDS = STOP_WORDS - IMPORTANT_WORDS

# ──────────────────────────────────────────────────────────────────
# Step 1 — Boilerplate removal
# ──────────────────────────────────────────────────────────────────


def remove_residual_boilerplate(text):
    """
    Even after BeautifulSoup stripping, raw text can contain
    leftover HTML entities (e.g. &amp;), stray inline JS
    snippets (e.g. var x = ...), and CSS fragments.
    This function catches those leftovers.
    """
    text = text.replace("&amp;", "and")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", " ")
    text = text.replace("&gt;", " ")
    text = text.replace("&quot;", " ")

    # strip anything that looks like inline JS or CSS blocks
    # e.g.  var x = 10; function foo() { ... }
    text = re.sub(r"var\s+\w+\s*=.*?;", " ", text)
    text = re.sub(r"function\s*\w*\s*\(.*?\)\s*\{.*?\}", " ", text, flags=re.DOTALL)
    # remove leftover CSS-like fragments  .class { ... }
    text = re.sub(r"\.\w+\s*\{[^}]*\}", " ", text)

    # remove any stray Hindi characters
    text = re.sub(r"[\u0900-\u097F]+", " ", text)

    # remove the IITJ website redirect tokens that sometimes bleed through
    text = re.sub(r"###.*?!!!", " ", text)

    # ---- Faculty page structured noise ----
    # Faculty pages repeat: "email user[at]iitj[dot]ac[dot]in call 0291 280 XXXX school"
    # Strip obfuscated email blocks:  word[at]word[dot]word[dot]word...
    text = re.sub(r"\w+\[at\]\w+(?:\[dot\]\w+)+", " ", text)
    # Strip "email <identifier>" and "call <phone>" labels
    text = re.sub(r"\bemail\s+\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcall\s+[\d\s\-\+]+", " ", text)
    # Strip "Ph.D." variations (common in faculty bios) — normalise to "phd"
    text = re.sub(r"\bPh\s*\.\s*D\s*\.?\s*:?", "phd", text, flags=re.IGNORECASE)

    # collapse duplicate lines (some pages repeat the same sentence)
    seen_lines = set()
    unique_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped not in seen_lines:
            seen_lines.add(stripped)
            unique_lines.append(stripped)
    text = "\n".join(unique_lines)

    return text


# ──────────────────────────────────────────────────────────────────
# Step 2 — Text cleaning
# ──────────────────────────────────────────────────────────────────


def clean_text(text):
    """
    Normalises and sanitises a block of text:
      - lowercasing for vocabulary consistency
      - removing URLs (http/https/www links)
      - removing email addresses
      - removing standalone numbers
      - removing excess punctuation and non-alpha noise
      - collapsing multiple spaces into one
    """
    # lowercase everything first
    text = text.lower()

    # remove URLs  (http, https, ftp, or bare www.…)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # remove email addresses (standard form and "at" / "dot" obfuscated)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # also catch display-obfuscated emails: name [at] domain [dot] ac [dot] in
    text = re.sub(
        r"\b\w+\s*(?:\[at\]|@)\s*\w+\s*(?:\[dot\]|\.)\s*\w+(?:\s*(?:\[dot\]|\.)\s*\w+)*",
        " ",
        text,
    )

    # remove "email:" / "ph:" / "call:" / "fax:" labels and any
    # trailing identifier they prefix (these litter faculty pages)
    text = re.sub(r"\b(?:email|e-mail|ph|phone|call|fax|tel)\s*[:]\s*\S*", " ", text)

    # remove domain-like fragments that survive after email stripping
    # e.g. "iitj.ac.in", "iitj dot ac dot in"
    text = re.sub(r"\biitj\s*(?:dot|\.)\s*ac\s*(?:dot|\.)\s*in\b", " ", text)

    # remove phone-number-like patterns (sequences of digits with
    # optional dashes or spaces in between)
    text = re.sub(r"[\+]?[\d][\d\s\-]{6,}\d", " ", text)

    # remove standalone numbers and mixed digit-punctuation chunks
    # but keep words that happen to contain digits (handled in filtering)
    text = re.sub(r"\b\d+\b", " ", text)

    # remove common punctuation that doesn't carry meaning
    # keep hyphens inside words (e.g. "state-of-the-art") for now;
    # isolated punctuation is removed
    text = re.sub(r"[^a-z\s\-]", " ", text)

    # collapse isolated hyphens that are not joining two words
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)

    # normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────────────────────────
# Step 3 — Tokenization
# ──────────────────────────────────────────────────────────────────


def tokenize_text(text):
    """
    Split the cleaned text into individual word tokens.
    We use a straightforward regex that matches sequences of
    alphabetic characters (possibly hyphenated).
    No heavy NLP library needed here — just regex splitting.
    """
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", text)
    return tokens


# ──────────────────────────────────────────────────────────────────
# Step 4 — Token filtering
# ──────────────────────────────────────────────────────────────────


def filter_tokens(tokens):
    filtered = []

    for tok in tokens:
        # length check
        if len(tok) < 2:
            continue

        # digit check
        if re.search(r"\d", tok):
            continue

        # punctuation check
        if re.fullmatch(r"[\-]+", tok):
            continue

        # 🚀 NEW: stopword removal
        if tok in STOP_WORDS:
            continue

        # 🚀 NEW: lemmatization
        tok = lemmatizer.lemmatize(tok)

        filtered.append(tok)

    return filtered


# ──────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────


def preprocess_document(raw_text):
    """
    Run the complete four-step pipeline on a single document's
    raw text and return a list of clean tokens.
    """
    text = remove_residual_boilerplate(raw_text)
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = filter_tokens(tokens)
    return tokens


def build_corpus(documents):
    """
    Process every scraped document through the pipeline.

    Parameters
    ----------
    documents : list[tuple[str, str]]
        Each element is (label, raw_text) as returned by the
        scraper module.

    Returns
    -------
    corpus_tokens : list[list[str]]
        A list of tokenized documents — exactly the format
        required for Word2Vec training later.
    """
    corpus_tokens = []
    for label, raw_text in documents:
        tokens = preprocess_document(raw_text)
        if len(tokens) > 0:
            corpus_tokens.append(tokens)
            print(f"  {label:30s} → {len(tokens):>5d} tokens")
        else:
            print(f"  {label:30s} → [empty after cleaning, skipped]")
    return corpus_tokens


# ──────────────────────────────────────────────────────────────────
# Saving the outputs
# ──────────────────────────────────────────────────────────────────


def save_clean_corpus(corpus_tokens, filepath):
    """
    Write the corpus to a plain-text file where each line is one
    document's tokens separated by spaces.

    Format:
        token1 token2 token3
        token4 token5 token6
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fout:
        for doc_tokens in corpus_tokens:
            line = " ".join(doc_tokens)
            fout.write(line + "\n")
    print(f"Clean corpus saved to {filepath}")


def save_corpus_pickle(corpus_tokens, filepath):
    """
    Serialize the tokenized corpus as a pickle file so it can
    be loaded directly for Word2Vec training without re-processing.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as fout:
        pickle.dump(corpus_tokens, fout)
    print(f"Tokenized corpus pickle saved to {filepath}")
