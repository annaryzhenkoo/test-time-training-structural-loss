import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        emb = self.embedding(input_ids)  # (B, T, E)
        outputs, hidden = self.gru(emb, hidden)  # (B, T, H)
        logits = self.fc(outputs)  # (B, T, V)
        return hidden, logits