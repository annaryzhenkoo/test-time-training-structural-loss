import torch
import random
from src.data.datasets_seq2seq import *
from src.train.adapt_ttt import adapt_ttt_for_one_example
import copy

@torch.no_grad()
def show_predictions(model, vocab: Vocab, representation="binary", device="cpu", n=10, max_len=32,
                     dataset = None, num_digits: int = 3):
    model.eval()
    if dataset is None:
        vocab = build_vocab(representation)
        dataset = AdditionDataset(
            vocab=vocab,
            num_samples=500,
            num_digits=num_digits,
            representation=representation
        )

    indices = random.sample(range(len(dataset)), k=min(n, len(dataset)))

    print("\nExamples WITHOUT test-time TTT:")
    for idx in indices:
        item = dataset[idx]
        src_ids = item["src_ids"].unsqueeze(0).to(device)
        src_lens = torch.tensor([len(item["src_ids"])], dtype=torch.long, device=device)

        pred_ids = model.greedy_decode(
            src_ids,
            src_lens,
            max_len=max_len,
            device=device
        )
        pred_tokens = decode_tokens(pred_ids, vocab)
        pred_text = "".join(pred_tokens)
        pred_value = parse_tokens_to_int(pred_tokens, representation)

        true_text = item["tgt_text"]
        true_value = item["sum_"]

        print("-" * 60)
        print(f"a = {item['a']}, b = {item['b']}")
        print(f"input (reversed):   {item['src_text']}")
        print(f"target (reversed):  {true_text}")
        print(f"pred   (reversed):  {pred_text}")
        print(f"true sum: {true_value}, pred sum: {pred_value}")

def show_predictions_with_ttt(
    model,
    dataset,
    vocab: Vocab,
    representation="binary",
    device="cpu",
    n=10,
    max_len=32,
    ttt_steps=5,
    ttt_lr=1e-2,
):
    model.eval()
    indices = random.sample(range(len(dataset)), k=min(n, len(dataset)))

    base_ttt_state = copy.deepcopy(model.ttt.state_dict())

    print("\nExamples WITH test-time TTT:")
    for idx in indices:
        item = dataset[idx]

        # reset TTT for this example
        model.ttt.load_state_dict(base_ttt_state)

        adapted_hidden = adapt_ttt_for_one_example(
            model=model,
            a=item["a"],
            b=item["b"],
            vocab=vocab,
            representation=representation,
            device=device,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
        )

        pred_ids = model.greedy_decode_from_hidden(
            adapted_hidden,
            max_len=max_len,
            device=device
        )
        pred_tokens = decode_tokens(pred_ids, vocab)
        pred_text = "".join(pred_tokens)
        pred_value = parse_tokens_to_int(pred_tokens, representation)

        true_text = item["tgt_text"]
        true_value = item["sum_"]

        print("-" * 60)
        print(f"a = {item['a']}, b = {item['b']}")
        print(f"input ab (reversed): {item['src_text']}")
        _, _, ba_text = build_src_ids_from_numbers(item["b"], item["a"], vocab, representation)
        print(f"input ba (reversed): {ba_text}")
        print(f"target (reversed):   {true_text}")
        print(f"pred   (reversed):   {pred_text}")
        print(f"true sum: {true_value}, pred sum: {pred_value}")

    model.ttt.load_state_dict(base_ttt_state)