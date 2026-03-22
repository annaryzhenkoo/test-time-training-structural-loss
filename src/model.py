import torch
import torch.nn as nn

class RNNAddition(nn.Module):

    def __init__(self, vocab_size:int, hidden_size:int, binary:bool):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.W = nn.Linear(2*hidden_size,hidden_size)
        self.O = nn.Linear(hidden_size, vocab_size)
        self.binary = binary

    def forward(self, inputs, h_0 = None):
        batch_size, seq_len = inputs.shape
        outputs = []
        inputs = self.emb_layer(inputs)

        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device = inputs.device)
        else:
            h_t = h_0

        for i in range(seq_len):
            x_t = inputs[:,i,:]
            h_t = torch.tanh(self.W(torch.cat([x_t, h_t], dim = -1)))
            output = self.O(h_t)
            outputs.append(output)

        return h_t, torch.stack(outputs, dim= 1)


class CharGRU(nn.Module):
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
