import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Weight matrices for the RNN recurrence
        self.Wxh = nn.Linear(input_size, hidden_size)  # input to hidden
        self.Whh = nn.Linear(hidden_size, hidden_size)  # hidden to hidden

    def forward(self, x_t, h_prev):

        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        h_t = torch.tanh(self.Wxh(x_t) + self.Whh(h_prev))
        return h_t


class VanillaRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Stack of RNN cells for multi-layer support
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(input_dim, hidden_size))

        # Output projection: hidden state -> vocabulary logits
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):

        batch_size, seq_len = x.size()

        # Initialize hidden states to zeros if not provided
        if hidden is None:
            hidden = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]

        # Embed input characters
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Process sequence step by step, maintaining hidden state across time
        outputs = []
        for t in range(seq_len):
            inp = embedded[:, t, :]  # (batch, embedding_dim)
            for layer_idx, cell in enumerate(self.rnn_cells):
                hidden[layer_idx] = cell(inp, hidden[layer_idx])
                inp = hidden[layer_idx]  # output of this layer feeds next layer
            outputs.append(inp)

        # Stack time-step outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)

        # Project to vocabulary size
        logits = self.output_layer(outputs)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def generate(self, vocab, start_char=None, max_len=20, temperature=1.0):

        self.eval()
        device = next(self.parameters()).device

        # Begin with <START> token
        input_idx = torch.tensor([[vocab.start_idx]], device=device)
        hidden = None
        generated_indices = []

        # If a starting character is provided, feed it first
        if start_char is not None:
            logits, hidden = self.forward(input_idx, hidden)
            char_idx = vocab.char_to_index[start_char]
            generated_indices.append(char_idx)
            input_idx = torch.tensor([[char_idx]], device=device)

        for _ in range(max_len):
            logits, hidden = self.forward(input_idx, hidden)
            logits = logits[:, -1, :] / temperature  # last time step

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab.end_idx:
                break

            generated_indices.append(next_idx)
            input_idx = torch.tensor([[next_idx]], device=device)

        return vocab.decode(generated_indices)
