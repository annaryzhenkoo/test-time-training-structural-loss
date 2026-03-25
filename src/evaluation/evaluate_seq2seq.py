from src.evaluation.metrics_seq2seq import exact_match_accuracy, exact_match_accuracy_with_ttt
from src.data.collate_seq2seq import *
from src.data.vocab import *
from torch.utils.data import DataLoader

def evaluation(num_digits, model, vocab: Vocab, device="cpu",
               num_samples: int = 2000, representation="binary"):


    dataset = AdditionDataset(
        vocab=vocab,
        num_samples=num_samples,
        num_digits=num_digits,
        representation=representation,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=make_collate_fn(vocab))

    test_exact_bin = exact_match_accuracy(
        model,
        loader,
        vocab=vocab,
        representation="binary",
        device=device,
        max_decode_len=140
    )
    print("Exact match without TTT:", test_exact_bin)

    test_exact_bin_ttt = exact_match_accuracy_with_ttt(
        model,
        loader,
        vocab=vocab,
        representation="binary",
        device=device,
        max_decode_len=140,
        ttt_steps=5,
        ttt_lr=1e-2,
    )
    print("Exact match with TTT:", test_exact_bin_ttt)