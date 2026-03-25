import os
import sys
import torch

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.vocabulary import CharVocabulary
from utils.dataset import NameDataset, load_names
from models.vanilla_rnn import VanillaRNN
from models.blstm import BLSTM
from models.attention_rnn import AttentionRNN
from train.train_rnn import train_vanilla_rnn
from train.train_blstm import train_blstm
from train.train_attention import train_attention_rnn

# Hyperparameters
EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 32

DATA_PATH = os.path.join(BASE_DIR, "data", "TrainingNames.txt")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_architecture(name, model, vocab_size):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"  Embedding Layer : vocab_size={vocab_size} -> {EMBEDDING_DIM}")
    print(f"  Hidden Size     : {HIDDEN_SIZE}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    print(f"  Learning Rate   : {LEARNING_RATE}")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Epochs          : {EPOCHS}")

    if "RNN" in name and "Attention" not in name and "BLSTM" not in name:
        print(f"  Activation      : tanh")
        print(
            f"  Architecture    : Embedding -> VanillaRNNCell x{NUM_LAYERS} -> Linear"
        )
    elif "BLSTM" in name:
        print(f"  Activation      : sigmoid (gates) + tanh (state)")
        print(
            f"  Architecture    : Embedding -> ForwardLSTM + BackwardLSTM -> Concat -> Linear"
        )
    elif "Attention" in name:
        print(f"  Activation      : tanh (RNN) + softmax (attention)")
        print(
            f"  Architecture    : Embedding -> RNNCell x{NUM_LAYERS} -> Attention -> Linear"
        )

    num_params = count_parameters(model)
    print(f"  Trainable Params: {num_params:,}")
    print(f"{'='*60}")
    return num_params


def main():
    print("=" * 60)
    print("Task-1: Character-Level Indian Name Generation")
    print("=" * 60)

    # Step 1: Load data and build vocabulary
    print("\n[Step 1] Loading dataset and building character vocabulary...")
    names = load_names(DATA_PATH)
    print(f"  Loaded {len(names)} names from dataset")

    vocab = CharVocabulary().build(names)
    print(f"  Vocabulary size: {vocab.vocab_size}")
    print(f"  Characters: {list(vocab.char_to_index.keys())}")
    print(
        f"  char_to_index mapping (first 10): {dict(list(vocab.char_to_index.items())[:10])}"
    )

    # Step 2: Create dataset─
    print("\n[Step 2] Encoding names into character sequences...")
    dataset = NameDataset(names, vocab)
    print(f"  Dataset size: {len(dataset)} sequences")

    # Show encoding example
    example_name = names[0]
    example_encoded = vocab.encode(example_name)
    tokens = [vocab.index_to_char[i] for i in example_encoded]
    print(f"  Example: '{example_name}' -> {tokens}")
    print(f"  Indices: {example_encoded}")

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Step 3: Vanilla RNN
    print("\n" + "#" * 60)
    print("# Training Vanilla RNN")
    print("#" * 60)

    rnn_model = VanillaRNN(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    rnn_params = print_architecture("Vanilla RNN", rnn_model, vocab.vocab_size)

    rnn_losses = train_vanilla_rnn(
        rnn_model,
        dataset,
        vocab,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=os.path.join(SAVE_DIR, "vanilla_rnn.pt"),
    )

    # Step 4: BLSTM
    print("\n" + "#" * 60)
    print("# Training Bidirectional LSTM (BLSTM)")
    print("#" * 60)

    blstm_model = BLSTM(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    blstm_params = print_architecture("BLSTM", blstm_model, vocab.vocab_size)

    blstm_losses = train_blstm(
        blstm_model,
        dataset,
        vocab,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=os.path.join(SAVE_DIR, "blstm.pt"),
    )

    # Step 5: RNN + Attention
    print("\n" + "#" * 60)
    print("# Training RNN with Attention")
    print("#" * 60)

    attn_model = AttentionRNN(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    attn_params = print_architecture("RNN + Attention", attn_model, vocab.vocab_size)

    attn_losses = train_attention_rnn(
        attn_model,
        dataset,
        vocab,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=os.path.join(SAVE_DIR, "attention_rnn.pt"),
    )

    # Summary─
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\nHyperparameters:")
    print(f"  embedding_dim  = {EMBEDDING_DIM}")
    print(f"  hidden_size    = {HIDDEN_SIZE}")
    print(f"  num_layers     = {NUM_LAYERS}")
    print(f"  learning_rate  = {LEARNING_RATE}")
    print(f"  epochs         = {EPOCHS}")
    print(f"  batch_size     = {BATCH_SIZE}")

    print(f"\nTrainable Parameters:")
    print(f"  Vanilla RNN       : {rnn_params:>10,}")
    print(f"  BLSTM             : {blstm_params:>10,}")
    print(f"  RNN + Attention   : {attn_params:>10,}")

    print(f"\nFinal Training Loss:")
    print(f"  Vanilla RNN       : {rnn_losses[-1]:.4f}")
    print(f"  BLSTM             : {blstm_losses[-1]:.4f}")
    print(f"  RNN + Attention   : {attn_losses[-1]:.4f}")

    # Generate sample names from each model─
    print(f"\nSample Generated Names:")
    print(f"  {'Model':<20} {'No prompt':<15} {'Start=A':<15} {'Start=S':<15}")
    print(f"  {'-'*65}")

    for model_name, model in [
        ("Vanilla RNN", rnn_model),
        ("BLSTM", blstm_model),
        ("RNN + Attention", attn_model),
    ]:
        model.eval()
        s1 = model.generate(vocab, temperature=0.8)
        s2 = model.generate(vocab, start_char="A", temperature=0.8)
        s3 = model.generate(vocab, start_char="S", temperature=0.8)
        print(f"  {model_name:<20} {s1:<15} {s2:<15} {s3:<15}")

    print(f"\nModels saved to: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
