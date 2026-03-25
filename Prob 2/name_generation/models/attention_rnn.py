import torch
import torch.nn as nn


class RNNCellManual(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.Wxh(x_t) + self.Whh(h_prev))


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        # Bilinear attention weight matrix
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h_current, all_hidden):

        # Transform current hidden state: (batch, hidden_size)
        query = self.W_a(h_current)  # (batch, hidden_size)

        # Attention scores: score_i = h_t^T W_a h_i
        # query unsqueezed: (batch, hidden_size, 1)
        # all_hidden: (batch, num_steps, hidden_size)
        scores = torch.bmm(all_hidden, query.unsqueeze(2)).squeeze(
            2
        )  # (batch, num_steps)

        # Normalize scores with softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (batch, num_steps)

        # Weighted sum of hidden states to form context vector
        context = torch.bmm(attn_weights.unsqueeze(1), all_hidden).squeeze(
            1
        )  # (batch, hidden_size)

        return context, attn_weights


class AttentionRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Stack of RNN cells
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim if i == 0 else hidden_size
            self.rnn_cells.append(RNNCellManual(input_dim, hidden_size))

        # Attention layer operates on the top-layer hidden states
        self.attention = Attention(hidden_size)

        # Output layer takes concatenation of context and hidden state
        self.output_layer = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, x, hidden=None):

        batch_size, seq_len = x.size()

        if hidden is None:
            hidden = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]

        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Collect all top-layer hidden states for attention
        all_hidden = []
        outputs = []

        for t in range(seq_len):
            inp = embedded[:, t, :]
            for layer_idx, cell in enumerate(self.rnn_cells):
                hidden[layer_idx] = cell(inp, hidden[layer_idx])
                inp = hidden[layer_idx]

            # Store the top-layer hidden state
            all_hidden.append(hidden[-1])

            # Stack all hidden states collected so far: (batch, t+1, hidden_size)
            H = torch.stack(all_hidden, dim=1)

            # Attention allows the model to focus on relevant previously-seen characters
            context, _ = self.attention(hidden[-1], H)

            # Combine context with current hidden state for richer representation
            combined = torch.cat([hidden[-1], context], dim=1)  # (batch, 2*hidden_size)
            outputs.append(combined)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, 2*hidden_size)
        logits = self.output_layer(outputs)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def generate(self, vocab, start_char=None, max_len=20, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device

        input_idx = torch.tensor([[vocab.start_idx]], device=device)
        hidden = None
        generated_indices = []
        all_hidden = []

        if start_char is not None:
            # Process <START> token
            hidden = [
                torch.zeros(1, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ]
            emb = self.embedding(input_idx).squeeze(1)
            inp = emb
            for i, cell in enumerate(self.rnn_cells):
                hidden[i] = cell(inp, hidden[i])
                inp = hidden[i]
            all_hidden.append(hidden[-1])

            char_idx = vocab.char_to_index[start_char]
            generated_indices.append(char_idx)
            input_idx = torch.tensor([[char_idx]], device=device)

        for _ in range(max_len):
            if hidden is None:
                hidden = [
                    torch.zeros(1, self.hidden_size, device=device)
                    for _ in range(self.num_layers)
                ]

            emb = self.embedding(input_idx).squeeze(1)
            inp = emb
            for i, cell in enumerate(self.rnn_cells):
                hidden[i] = cell(inp, hidden[i])
                inp = hidden[i]

            all_hidden.append(hidden[-1])
            H = torch.stack(all_hidden, dim=1)
            context, _ = self.attention(hidden[-1], H)
            combined = torch.cat([hidden[-1], context], dim=1)

            logits = self.output_layer(combined) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab.end_idx:
                break

            generated_indices.append(next_idx)
            input_idx = torch.tensor([[next_idx]], device=device)

        return vocab.decode(generated_indices)
