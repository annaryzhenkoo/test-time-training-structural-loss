import torch
import torch.nn as nn
from src.data.vocab import Vocab

class Encoder(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab.VOCAB_SIZE, emb_dim, padding_idx=vocab.PAD_ID)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, src_ids, src_lens):
        emb = self.embedding(src_ids)       # (B, S, E)
        outputs, _ = self.gru(emb)          # (B, S, H)

        batch_size = src_ids.size(0)
        batch_indices = torch.arange(batch_size, device=src_ids.device)
        last_token_indices = src_lens - 1 # last not padding

        last_outputs = outputs[batch_indices, last_token_indices, :]  # (B, H)
        hidden = last_outputs.unsqueeze(0)                            # (1, B, H)

        return outputs, hidden


class TTTLayer(nn.Module):
    """
    residual adaptation hidden state:
        z = h + W2(ReLU(W1(h)))
    (1, B, H)
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, hidden):
        # hidden: (1, B, H)
        x = hidden.squeeze(0)          # (B, H)
        delta = self.net(x)            # (B, H)
        z = x + delta                  # residual
        return z.unsqueeze(0)          # (1, B, H)


class Decoder(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab.VOCAB_SIZE, emb_dim, padding_idx=vocab.PAD_ID)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab.VOCAB_SIZE)

    def forward(self, tgt_input_ids, hidden):
        emb = self.embedding(tgt_input_ids)     # (B, T, E)
        outputs, hidden = self.gru(emb, hidden) # (B, T, H)
        logits = self.fc(outputs)               # (B, T, V)
        return logits, hidden

class Seq2SeqGRUWithTTT(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim=32, hidden_dim=128, ttt_bottleneck_dim=None):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(vocab, emb_dim, hidden_dim)
        self.ttt = TTTLayer(hidden_dim, bottleneck_dim=ttt_bottleneck_dim)
        self.decoder = Decoder(vocab, emb_dim, hidden_dim)

    def encode(self, src_ids, src_lens):
        _, hidden = self.encoder(src_ids, src_lens)
        return hidden

    def forward(self, src_ids, src_lens, tgt_input_ids):
        hidden = self.encode(src_ids, src_lens)   # (1, B, H)
        hidden = self.ttt(hidden)                 # (1, B, H)
        logits, _ = self.decoder(tgt_input_ids, hidden)
        return logits

    @torch.no_grad()
    def greedy_decode_from_hidden(self, hidden, max_len=32, device="cpu"):
        self.eval()

        current = torch.tensor([[self.vocab.SOS_ID]], dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            logits, hidden = self.decoder(current, hidden)   # (1, 1, V)
            next_token = logits[:, -1, :].argmax(dim=-1)     # (1,)
            token_id = int(next_token.item())

            if token_id == self.vocab.EOS_ID:
                break

            generated.append(token_id)
            current = next_token.unsqueeze(1)

        return generated

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_lens, max_len=32, device="cpu"):
        self.eval()
        hidden = self.encode(src_ids.to(device), src_lens.to(device))
        hidden = self.ttt(hidden)
        return self.greedy_decode_from_hidden(hidden, max_len=max_len, device=device)
