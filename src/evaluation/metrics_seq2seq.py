from src.models.seq2seq_gru import Seq2SeqGRUWithTTT
from src.data.datasets_seq2seq import *
from src.train.adapt_ttt import adapt_ttt_for_one_example
import copy

@torch.no_grad()
def exact_match_accuracy(model, dataloader, vocab: Vocab, representation="binary", device="cpu", max_decode_len=32):
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        src_lens = batch["src_lens"].to(device)
        sums = batch["sum_"]

        for i in range(src_ids.size(0)):
            one_src = src_ids[i:i+1]
            one_len = src_lens[i:i+1]

            pred_ids = model.greedy_decode(
                one_src,
                one_len,
                max_len=max_decode_len,
                device=device
            )
            pred_tokens = decode_tokens(pred_ids, vocab)
            pred_value = parse_tokens_to_int(pred_tokens, representation)

            if pred_value == sums[i]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def exact_match_accuracy_with_ttt(
    model: Seq2SeqGRUWithTTT,
    dataloader,
    vocab: Vocab,
    representation="binary",
    device="cpu",
    max_decode_len=32,
    ttt_steps=5,
    ttt_lr=1e-2,
):
    """
    for each example adapt TTT, use hidden state for decoding
    reset TTT
    """
    model.eval()
    correct = 0
    total = 0

    base_ttt_state = copy.deepcopy(model.ttt.state_dict())

    for batch in dataloader:
        a_list = batch["a"]
        b_list = batch["b"]
        sums = batch["sum_"]

        for i in range(len(a_list)):
            # reset TTT before each example
            model.ttt.load_state_dict(base_ttt_state)

            adapted_hidden = adapt_ttt_for_one_example(
                model=model,
                a=a_list[i],
                b=b_list[i],
                vocab=vocab,
                representation=representation,
                device=device,
                ttt_steps=ttt_steps,
                ttt_lr=ttt_lr,
            )

            pred_ids = model.greedy_decode_from_hidden(
                adapted_hidden,
                max_len=max_decode_len,
                device=device
            )
            pred_tokens = decode_tokens(pred_ids, vocab)
            pred_value = parse_tokens_to_int(pred_tokens, representation)

            if pred_value == sums[i]:
                correct += 1
            total += 1

    # restore original TTT after eval
    model.ttt.load_state_dict(base_ttt_state)

    return correct / total if total > 0 else 0.0

def token_accuracy(logits, targets, pad_id):
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def get_value_range_by_num_digits(num_digits: int, representation: str):
    if num_digits < 1:
        raise ValueError("num_digits must be >= 1")

    if representation == "decimal":
        if num_digits == 1:
            return 0, 9
        return 10 ** (num_digits - 1), 10 ** num_digits - 1

    elif representation == "binary":
        if num_digits == 1:
            return 0, 1
        return 10 ** (num_digits - 1), 10 ** num_digits - 1

    else:
        raise ValueError(f"Unknown representation: {representation}")


def exact_match_accuracy_with_ttt_online(
    model: Seq2SeqGRUWithTTT,
    dataloader,
    vocab: Vocab,
    representation="binary",
    device="cpu",
    max_decode_len=32,
    ttt_steps=5,
    ttt_lr=1e-2,
    restore_after_eval=True,
):
    """
    TTT-online:
    - no reset before each example
    - TTT weights are updated sequentially across evaluation examples
    - after adapting on example i, updated TTT is used for example i+1

    If restore_after_eval=True, original TTT weights are restored at the end.
    """
    model.eval()
    correct = 0
    total = 0

    base_ttt_state = copy.deepcopy(model.ttt.state_dict())

    for batch in dataloader:
        a_list = batch["a"]
        b_list = batch["b"]
        sums = batch["sum_"]

        for i in range(len(a_list)):
            # IMPORTANT:
            # no reset here; TTT keeps evolving across examples

            adapted_hidden = adapt_ttt_for_one_example(
                model=model,
                a=a_list[i],
                b=b_list[i],
                vocab=vocab,
                representation=representation,
                device=device,
                ttt_steps=ttt_steps,
                ttt_lr=ttt_lr,
            )

            pred_ids = model.greedy_decode_from_hidden(
                adapted_hidden,
                max_len=max_decode_len,
                device=device
            )
            pred_tokens = decode_tokens(pred_ids, vocab)
            pred_value = parse_tokens_to_int(pred_tokens, representation)

            if pred_value == sums[i]:
                correct += 1
            total += 1

    if restore_after_eval:
        model.ttt.load_state_dict(base_ttt_state)

    return correct / total if total > 0 else 0.0