import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # Gate weight matrices: each gate has its own linear transformation
        self.W_f = nn.Linear(combined_size, hidden_size)  # forget gate
        self.W_i = nn.Linear(combined_size, hidden_size)  # input gate
        self.W_c = nn.Linear(combined_size, hidden_size)  # candidate cell
        self.W_o = nn.Linear(combined_size, hidden_size)  # output gate

    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state: [h_{t-1}, x_t]
        combined = torch.cat([h_prev, x_t], dim=1)

        # Compute all four gates
        f_t = torch.sigmoid(self.W_f(combined))  # forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # input gate
        c_tilde = torch.tanh(self.W_c(combined))  # candidate cell state
        o_t = torch.sigmoid(self.W_o(combined))  # output gate

        # Update cell state: selectively forget and add new information
        c_t = f_t * c_prev + i_t * c_tilde

        # Compute hidden state from cell state, gated by output gate
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class BLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Forward and backward LSTM cells for each layer
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes embedding input; subsequent layers take
            # concatenated bidirectional hidden states (2 * hidden_size)
            input_dim = embedding_dim if i == 0 else 2 * hidden_size
            self.forward_cells.append(LSTMCell(input_dim, hidden_size))
            self.backward_cells.append(LSTMCell(input_dim, hidden_size))

        # Output projection from concatenated bidirectional hidden states
        self.output_layer = nn.Linear(2 * hidden_size, vocab_size)

    def _run_direction(self, cells, embedded, reverse=False):

        batch_size, seq_len, _ = embedded.size()
        device = embedded.device

        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        layer_input = embedded
        for layer_idx, cell in enumerate(cells):
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            hidden_seq = []

            for t in time_steps:
                h, c = cell(layer_input[:, t, :], h, c)
                hidden_seq.append(h)

            # If processing was reversed, reverse the collected outputs
            # so they align with the original time ordering
            if reverse:
                hidden_seq = hidden_seq[::-1]

            layer_input = torch.stack(
                hidden_seq, dim=1
            )  # (batch, seq_len, hidden_size)

        return layer_input

    def forward(self, x, hidden=None):

        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Run forward and backward LSTMs
        h_forward = self._run_direction(self.forward_cells, embedded, reverse=False)
        h_backward = self._run_direction(self.backward_cells, embedded, reverse=True)

        # Concatenate forward and backward hidden states
        h_bi = torch.cat(
            [h_forward, h_backward], dim=2
        )  # (batch, seq_len, 2*hidden_size)

        logits = self.output_layer(h_bi)  # (batch, seq_len, vocab_size)
        return logits, None

    def generate(self, vocab, start_char=None, max_len=20, temperature=1.0):

        self.eval()
        device = next(self.parameters()).device

        input_idx = torch.tensor([[vocab.start_idx]], device=device)
        generated_indices = []

        # Initialize forward cell states
        h_states = [
            torch.zeros(1, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(1, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

        if start_char is not None:
            emb = self.embedding(input_idx).squeeze(1)
            inp = emb
            for i, cell in enumerate(self.forward_cells):
                h_states[i], c_states[i] = cell(inp, h_states[i], c_states[i])
                # For generation, duplicate forward hidden to fill the 2*hidden projection
                inp = torch.cat([h_states[i], h_states[i]], dim=1)

            char_idx = vocab.char_to_index[start_char]
            generated_indices.append(char_idx)
            input_idx = torch.tensor([[char_idx]], device=device)

        for _ in range(max_len):
            emb = self.embedding(input_idx).squeeze(1)  # (1, embedding_dim)
            inp = emb
            for i, cell in enumerate(self.forward_cells):
                h_states[i], c_states[i] = cell(inp, h_states[i], c_states[i])
                # Duplicate forward hidden to match the 2*hidden_size expected by next layer
                inp = torch.cat([h_states[i], h_states[i]], dim=1)

            # For output projection, create pseudo-bidirectional representation
            h_combined = torch.cat([h_states[-1], h_states[-1]], dim=1)
            logits = self.output_layer(h_combined) / temperature

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab.end_idx:
                break

            generated_indices.append(next_idx)
            input_idx = torch.tensor([[next_idx]], device=device)

        return vocab.decode(generated_indices)
