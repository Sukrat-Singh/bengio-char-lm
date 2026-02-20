import torch
import torch.nn as nn

class BengioLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, hidden_dim):
        super().__init__()

        # hyperparameters 
        self.V = vocab_size
        self.k = context_len
        self.m = embed_dim
        self.h = hidden_dim

        # ---- Layers ----
        # Matrix C (embedding matrix)
        self.embedding = nn.Embedding(self.V, self.m)

        # Hidden layer: Hx + b
        self.hidden_linear = nn.Linear(self.k * self.m, self.h)

        # Linear shortcut Wx (no bias)
        self.input_to_vocab = nn.Linear(self.k * self.m, self.V, bias=False)

        # Nonlinear path Uh (no bias)
        self.hidden_to_vocab = nn.Linear(self.h, self.V, bias=False)

        # Output bias b
        self.bias = nn.Parameter(torch.zeros(self.V))

    def forward(self, x):
        # x shape: [B, k]

        # 1. Embedding lookup → [B, k, m]
        emb = self.embedding(x)

        # 2. Flatten context → [B, k*m]
        B = emb.shape[0]
        x_flat = emb.view(B, -1)

        # 3. Hidden tanh layer
        hidden = torch.tanh(self.hidden_linear(x_flat))

        # 4. Bengio output: Wx + Uh + b
        logits = (
            self.input_to_vocab(x_flat)
            + self.hidden_to_vocab(hidden)
            + self.bias
        )

        return logits