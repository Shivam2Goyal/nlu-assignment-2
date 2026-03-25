import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_vanilla_rnn(
    model,
    dataset,
    vocab,
    epochs=25,
    batch_size=32,
    learning_rate=0.001,
    device="cpu",
    save_path=None,
):

    from utils.dataset import collate_fn

    model = model.to(device)
    model.train()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cross-entropy loss, ignoring padding tokens (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    training_losses = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass through the RNN
            logits, _ = model(inputs)  # (batch, seq_len, vocab_size)

            # Reshape for cross-entropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

            loss.backward()
            # Gradient clipping to prevent exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        training_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Generate a sample name to monitor training progress
            sample = model.generate(vocab, temperature=0.8)
            print(
                f"  Epoch [{epoch+1}/{epochs}] — Loss: {avg_loss:.4f} — Sample: {sample}"
            )

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved to {save_path}")

    return training_losses
