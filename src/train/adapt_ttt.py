import torch
from src.data.datasets_seq2seq import *
from src.models.seq2seq_gru import *

def adapt_ttt_for_one_example(
    model: Seq2SeqGRUWithTTT,
    a: int,
    b: int,
    vocab: Vocab,
    representation: str,
    device: str = "cpu",
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
):
    """
    For one example:
      1) build a+b и b+a
      2) encoder frozen, decoder frozen
      3) update only TTT to make TTT(h_ab) and TTT(h_ba) similar
      4) after adaptation return hidden for a+b
    """
    model.eval()

    # freeze encoder/decoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    src_ab, len_ab, _ = build_src_ids_from_numbers(a, b, vocab, representation)
    src_ba, len_ba, _ = build_src_ids_from_numbers(b, a, vocab, representation)

    src_ab = src_ab.to(device)
    len_ab = len_ab.to(device)
    src_ba = src_ba.to(device)
    len_ba = len_ba.to(device)

    with torch.no_grad():
        h_ab = model.encode(src_ab, len_ab)   # (1, 1, H)
        h_ba = model.encode(src_ba, len_ba)   # (1, 1, H)

    optimizer_ttt = torch.optim.SGD(model.ttt.parameters(), lr=ttt_lr)

    for _ in range(ttt_steps):
        optimizer_ttt.zero_grad()

        z_ab = model.ttt(h_ab)   # (1, 1, H)
        z_ba = model.ttt(h_ba)   # (1, 1, H)

        inner_loss = ((z_ab - z_ba) ** 2).mean()
        inner_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.ttt.parameters(), max_norm=1.0)
        optimizer_ttt.step()

    with torch.no_grad():
        adapted_hidden_ab = model.ttt(h_ab)

    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    return adapted_hidden_ab