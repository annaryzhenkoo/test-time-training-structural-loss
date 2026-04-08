import torch
import torch.nn as nn
from src.data.vocab import Vocab
import random

class Encoder(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab.VOCAB_SIZE, emb_dim, padding_idx=vocab.PAD_ID)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def build_place_ids(self, src_ids):
        B, S = src_ids.shape
        place_ids = torch.zeros_like(src_ids)

        plus_id = self.vocab.stoi["+"]
        eq_id = self.vocab.stoi["="]
        pad_id = self.vocab.PAD_ID

        for b in range(B):
            current_pos = 0
            for t in range(S):
                tok = src_ids[b, t].item()

                if tok == plus_id:
                    place_ids[b, t] = 0
                    current_pos = 0
                elif tok == eq_id or tok == pad_id:
                    place_ids[b, t] = 0
                else:
                    place_ids[b, t] = current_pos
                    current_pos += 1

        return place_ids

    def build_digit_mask(self, src_ids):
        plus_id = self.vocab.stoi["+"]
        eq_id = self.vocab.stoi["="]
        pad_id = self.vocab.PAD_ID

        return (src_ids != plus_id) & (src_ids != eq_id) & (src_ids != pad_id)

    def sinusoidal_embedding(self, place_ids):
        device = place_ids.device
        B, S = place_ids.shape
        place_ids = place_ids.float().unsqueeze(-1)  # (B, S, 1)

        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2, device=device).float()
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.emb_dim)
        )

        pe = torch.zeros(B, S, self.emb_dim, device=device)
        pe[:, :, 0::2] = torch.sin(place_ids * div_term)
        pe[:, :, 1::2] = torch.cos(place_ids * div_term)

        return pe

    def forward(self, src_ids, src_lens):
        token_emb = self.embedding(src_ids)   # (B, S, E)

        place_ids = self.build_place_ids(src_ids)         # (B, S)
        pos_emb = self.sinusoidal_embedding(place_ids)    # (B, S, E)

        digit_mask = self.build_digit_mask(src_ids).unsqueeze(-1)   # (B, S, 1)
        pos_emb = pos_emb * digit_mask

        emb = token_emb + pos_emb

        outputs, _ = self.gru(emb)

        batch_size = src_ids.size(0)
        batch_indices = torch.arange(batch_size, device=src_ids.device)
        last_token_indices = src_lens - 1

        last_outputs = outputs[batch_indices, last_token_indices, :]
        hidden = last_outputs.unsqueeze(0)

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

    def forward_scheduled_sampling(self, tgt_input_ids, hidden, current_p=1.0):
        B, T = tgt_input_ids.shape
        current_ids = tgt_input_ids[:, 0].unsqueeze(1)   # (B, 1)
        all_logits = []

        for t in range(T):
            emb = self.embedding(current_ids)  # (B, 1, E)
            outputs, hidden = self.gru(emb, hidden)  # (B, 1, H)
            logits = self.fc(outputs)  # (B, 1, V)
            all_logits.append(logits)

            pred_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

            if t + 1 < T:
                teacher_ids = tgt_input_ids[:, t + 1].unsqueeze(1)  # (B, 1)
                teacher_mask = torch.rand(B, 1, device=tgt_input_ids.device) < current_p
                current_ids = torch.where(teacher_mask, teacher_ids, pred_ids)

        all_logits = torch.cat(all_logits, dim=1)  # (B, T, V)
        return all_logits, hidden

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

    def forward(self, src_ids, src_lens, tgt_input_ids, scheduled_sampling: bool = False, current_p: float = 1.0):
        hidden = self.encode(src_ids, src_lens)  # (1, B, H)
        hidden = self.ttt(hidden)  # (1, B, H)

        if scheduled_sampling:
            logits, _ = self.decoder.forward_scheduled_sampling(
                tgt_input_ids, hidden, current_p=current_p
            )
        else:
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
