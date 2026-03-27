import torch
import torch.nn.functional as F
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
        inner_function: str = "commutative",
        similarity_loss: str = "normalized_l2",
):
    """
    Adapt only TTT block for one example using structural inner loss.

    Supported inner_function:
      1. "commutative"
         enforce TTT(h(a,b)) == TTT(h(b,a))

      2. "zero_commutativity"
         enforce TTT(h(a,0)) == TTT(h(0,a))

    Supported similarity_loss:
      1. "normalized_l2"
         normalize both vectors, then minimize L2 distance

      2. "cosine"
         maximize cosine similarity

      3. "smooth_l1"
         minimize smooth L1 distance

    Returns:
        adapted hidden state for the original pair (a, b)
    """
    model.eval()

    # freeze encoder/decoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    def encode_pair(x: int, y: int):
        src, src_len, _ = build_src_ids_from_numbers(x, y, vocab, representation)
        src = src.to(device)
        src_len = src_len.to(device)

        with torch.no_grad():
            h = model.encode(src, src_len)  # expected shape: (1, 1, H)
        return h

    def compute_similarity_loss(z_left, z_right, loss_type: str):
        if loss_type == "normalized_l2":
            z_left = F.normalize(z_left, dim=-1)
            z_right = F.normalize(z_right, dim=-1)
            return ((z_left - z_right) ** 2).mean()

        elif loss_type == "cosine":
            return 1.0 - F.cosine_similarity(z_left, z_right, dim=-1).mean()

        elif loss_type == "smooth_l1":
            return F.smooth_l1_loss(z_left, z_right)

        else:
            raise ValueError(
                f"Unknown similarity_loss='{loss_type}'. "
                f"Supported values: 'normalized_l2', 'cosine', 'smooth_l1'"
            )

    # original hidden for final decoding
    h_ab_original = encode_pair(a, b)

    if inner_function == "commutative":
        h_left = encode_pair(a, b)  # h_ab
        h_right = encode_pair(b, a)  # h_ba

    elif inner_function == "zero_commutativity":
        h_left = encode_pair(a, 0)  # h_a0
        h_right = encode_pair(0, a)  # h_0a

    else:
        raise ValueError(
            f"Unknown inner_function='{inner_function}'. "
            f"Supported values: 'commutative', 'zero_commutativity'"
        )

    optimizer_ttt = torch.optim.SGD(model.ttt.parameters(), lr=ttt_lr)

    for _ in range(ttt_steps):
        optimizer_ttt.zero_grad()

        z_left = model.ttt(h_left)
        z_right = model.ttt(h_right)

        inner_loss = compute_similarity_loss(z_left, z_right, similarity_loss)
        inner_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.ttt.parameters(), max_norm=1.0)
        optimizer_ttt.step()

    with torch.no_grad():
        adapted_hidden_ab = model.ttt(h_ab_original)

    # unfreeze encoder/decoder back
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    return adapted_hidden_ab