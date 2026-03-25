# CSL 7640: Natural Language Understanding — Assignment 2

**Name:** Shivam Goyal | **Roll No:** B23CM1036

This repository contains two end-to-end NLP implementations:

- **Problem 1:** Word2Vec (CBOW + Skip-Gram with Negative Sampling) trained from scratch on an IIT Jodhpur corpus
- **Problem 2:** Character-level Indian name generation using Vanilla RNN, Bidirectional LSTM, and RNN with Attention — all implemented from scratch

---

## Repository Structure

```
NLU_Ass2/
├── Prob 1/
│   ├── dataset_preparation/
│   │   ├── data/
│   │   │   ├── clean_corpus.txt          # Preprocessed corpus (output)
│   │   │   ├── corpus_tokens.pkl         # Tokenized corpus pickle
│   │   │   └── raw_pages/               # 47 scraped IITJ web pages (.txt)
│   │   ├── scripts/
│   │   │   ├── scraper.py               # Web scraper for IITJ pages
│   │   │   ├── preprocess.py            # Cleaning, tokenization, lemmatization
│   │   │   ├── statistics.py            # Corpus stats computation
│   │   │   └── wordcloud_visualization.py
│   │   └── main.py                      # Run full dataset pipeline
│   ├── word2vec_training/
│   │   ├── models/
│   │   │   ├── cbow.py                  # CBOW model (from scratch)
│   │   │   └── skipgram.py              # Skip-Gram model (from scratch)
│   │   ├── utils/
│   │   │   ├── vocabulary.py            # Vocab builder + unigram distribution
│   │   │   ├── dataset.py               # CBOW/SkipGram pair generators + Datasets
│   │   │   └── negative_sampling.py     # Negative sampler with lookup table
│   │   ├── data/
│   │   │   ├── clean_corpus.txt         # Corpus used for training
│   │   │   ├── trained_cbow_model.pt    # Saved best CBOW model
│   │   │   ├── trained_skipgram_model.pt
│   │   │   ├── word_embeddings.pkl      # Combined embeddings dict
│   │   │   └── training_results.csv     # Loss across all experiments
│   │   ├── train_cbow.py               # CBOW training function
│   │   ├── train_skipgram.py           # Skip-Gram training function
│   │   └── experiments.py              # Hyperparameter grid search (main entry)
│   ├── semantic_analysis/
│   │   ├── similarity.py               # Cosine similarity + nearest neighbors
│   │   ├── analogy.py                  # Vector arithmetic analogy solver
│   │   ├── evaluation.py               # Runs neighbor + analogy experiments
│   │   ├── load_embeddings.py          # Loads word_embeddings.pkl
│   │   ├── main.py                     # Run full semantic analysis
│   │   └── semantic_analysis_results.txt
│   └── visualization/
│       ├── pca_visualization.py
│       ├── tsne_visualization.py
│       ├── plot_utils.py
│       ├── load_embeddings.py
│       ├── main.py                     # Run PCA + t-SNE plots
│       └── visualizations/
│           ├── cbow_pca.png
│           ├── cbow_tsne.png
│           ├── skipgram_pca.png
│           └── skipgram_tsne.png
│
└── Prob 2/
    └── name_generation/
        ├── data/
        │   └── TrainingNames.txt        # 1000 Indian names
        ├── models/
        │   ├── vanilla_rnn.py           # Vanilla RNN (manual cell)
        │   ├── blstm.py                 # BLSTM (manual LSTM gates)
        │   └── attention_rnn.py         # RNN + Bilinear Attention
        ├── train/
        │   ├── train_rnn.py
        │   ├── train_blstm.py
        │   └── train_attention.py
        ├── utils/
        │   ├── vocabulary.py            # CharVocabulary (char ↔ index)
        │   └── dataset.py               # NameDataset + collate_fn
        ├── evaluation/
        │   ├── metrics.py               # Novelty + Diversity computation
        │   ├── generate_names.py        # Name generation with temperature
        │   ├── evaluation.py            # Runs full quantitative evaluation
        │   ├── qualitative_analysis.py  # Realism + failure mode analysis
        │   ├── main.py                  # Run full evaluation pipeline
        │   └── results/
        │       ├── evaluation_results.txt
        │       ├── qualitative_analysis.txt
        │       ├── rnn_generated.txt
        │       ├── blstm_generated.txt
        │       └── attention_generated.txt
        ├── saved_models/
        │   ├── vanilla_rnn.pt
        │   ├── blstm.pt
        │   └── attention_rnn.pt
        └── main.py                      # Train all 3 models (main entry)
```

---

## Requirements

```bash
pip install torch numpy nltk matplotlib scikit-learn wordcloud
```

Download NLTK data (required for Problem 1 preprocessing):

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Problem 1: Word2Vec

### Step 1 — Dataset Preparation

Preprocesses the raw scraped pages into a clean corpus with stopword removal and lemmatization.

```bash
cd "Prob 1/dataset_preparation"
python main.py
```

**Output:** `data/clean_corpus.txt`, `data/corpus_tokens.pkl`, `wordcloud.png`

---

### Step 2 — Train Word2Vec Models

Runs a hyperparameter grid search over embedding dimensions {50, 100, 200}, window sizes {2, 4, 6}, and negative sample counts {5, 10}.

```bash
cd "Prob 1/word2vec_training"
python experiments.py
```

**Output:**
- `data/trained_cbow_model.pt` — best CBOW model weights
- `data/trained_skipgram_model.pt` — best Skip-Gram model weights
- `data/word_embeddings.pkl` — combined embeddings dictionary
- `data/training_results.csv` — loss table for all 27 runs

To train a single model with custom hyperparameters:

```bash
# CBOW
python train_cbow.py

# Skip-Gram
python train_skipgram.py
```

---

### Step 3 — Semantic Analysis

Computes nearest neighbors (cosine similarity) and runs analogy experiments on both CBOW and Skip-Gram embeddings.

```bash
cd "Prob 1"
python semantic_analysis/main.py
```

**Output:** `semantic_analysis/semantic_analysis_results.txt`

---

### Step 4 — Visualization (PCA + t-SNE)

Projects word embeddings into 2D and generates cluster plots.

```bash
cd "Prob 1"
python visualization/main.py
```

**Output:** `visualization/visualizations/` — 4 PNG files (CBOW/Skip-Gram × PCA/t-SNE)

---

### Problem 1 — Quick Run (All Steps)

```bash
cd "Prob 1/word2vec_training" && python experiments.py
cd "../.."
python "Prob 1/semantic_analysis/main.py"
python "Prob 1/visualization/main.py"
```

---

## Problem 2: Character-Level Name Generation

### Step 1 — Train All Models

Trains Vanilla RNN, BLSTM, and RNN with Attention sequentially. Prints loss per epoch and sample generated names during training.

```bash
cd "Prob 2/name_generation"
python main.py
```

**Output:**
- `saved_models/vanilla_rnn.pt`
- `saved_models/blstm.pt`
- `saved_models/attention_rnn.pt`

**Hyperparameters used:**

| Parameter | Value |
|---|---|
| Embedding dim | 32 |
| Hidden size | 128 |
| Layers | 1 |
| Epochs | 25 |
| Batch size | 32 |
| Learning rate | 0.001 (Adam) |
| Gradient clip | max norm 5.0 |

---

### Step 2 — Evaluate Models

Generates 1000 names per model and computes Novelty Rate and Diversity.

```bash
cd "Prob 2/name_generation"
python evaluation/main.py
```

**Output:**
- `evaluation/results/evaluation_results.txt` — quantitative metrics
- `evaluation/results/qualitative_analysis.txt` — realism + failure mode analysis
- `evaluation/results/rnn_generated.txt`
- `evaluation/results/blstm_generated.txt`
- `evaluation/results/attention_generated.txt`

---

### Step 2b — Generate Names Interactively

You can generate names from a saved model with a custom starting character or temperature:

```python
import torch
from utils.vocabulary import CharVocabulary
from utils.dataset import load_names
from models.attention_rnn import AttentionRNN

names = load_names("data/TrainingNames.txt")
vocab = CharVocabulary().build(names)

model = AttentionRNN(vocab.vocab_size, embedding_dim=32, hidden_size=128)
model.load_state_dict(torch.load("saved_models/attention_rnn.pt"))

# Generate 10 names
for _ in range(10):
    print(model.generate(vocab, temperature=0.8))

# Generate names starting with 'A'
for _ in range(10):
    print(model.generate(vocab, start_char='A', temperature=0.8))
```

---

### Problem 2 — Quick Run (All Steps)

```bash
cd "Prob 2/name_generation"
python main.py          # Train all 3 models
python evaluation/main.py   # Evaluate + generate results
```

---

## Key Results

### Problem 1 — Word2Vec

| Model | Best Config | Final Loss |
|---|---|---|
| CBOW | dim=300, window=2 | **1.2966** |
| Skip-Gram | dim=300, window=2, K=5 | 1.3444 |

CBOW outperforms Skip-Gram on this small corpus. Smaller window sizes (2) and larger embedding dims (300) consistently give better results.

### Problem 2 — Name Generation

| Model | Params | Novelty | Diversity | Indian Endings |
|---|---|---|---|---|
| Vanilla RNN | 28,786 | 0.114 | 0.644 | 77.9% |
| BLSTM | 179,314 | 0.997 | 0.933 | 6.1% |
| **RNN + Attention** | **51,570** | **0.136** | **0.644** | **75.5%** |

**RNN + Attention** produces the most realistic names. BLSTM's high novelty is a generation failure artefact (bidirectional train/generate mismatch), not a quality indicator.

---

## Implementation Notes

- **No library shortcuts:** All RNN cells, LSTM gates, CBOW/Skip-Gram forward passes, and the attention mechanism are implemented manually. `nn.RNN`, `nn.LSTM`, `nn.GRU`, and Gensim's Word2Vec are not used anywhere.
- **Negative Sampling:** Uses a pre-built lookup table of size 10⁶ for O(1) sampling from the smoothed unigram distribution P(w) ∝ f(w)^0.75.
- **Gradient Clipping:** Applied in all RNN training loops (max norm = 5.0) to prevent exploding gradients.
- **Temperature Sampling:** All name generation uses temperature τ = 0.8 for a balance between diversity and phonetic coherence.
