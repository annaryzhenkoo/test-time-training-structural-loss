import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.datasets_seq2seq import AdditionDataset
from src.data.collate_seq2seq import make_collate_fn
from src.data.vocab import Vocab
from src.evaluation.metrics_seq2seq import (
    token_accuracy,
    exact_match_accuracy,
    exact_match_accuracy_with_ttt,
    exact_match_accuracy_with_ttt_online,
)
from src.train.adapt_ttt import adapt_ttt_for_one_example


def evaluation(
    num_digits: int,
    model,
    vocab: Vocab,
    device: str = "cpu",
    num_samples: int = 2000,
    representation: str = "binary",
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
    batch_size: int = 128,
    max_decode_len: int = 140,
    inner_loss: str = "commutative"
):
    print(f"Evaluation on {num_digits} digits")

    dataset = AdditionDataset(
        vocab=vocab,
        num_samples=num_samples,
        num_digits=num_digits,
        representation=representation,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(vocab),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

    model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            src_lens = batch["src_lens"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            tgt_output_ids = batch["tgt_output_ids"].to(device)

            logits = model(src_ids, src_lens, tgt_input_ids)

            loss = criterion(
                logits.reshape(-1, vocab.VOCAB_SIZE),
                tgt_output_ids.reshape(-1)
            )

            acc = token_accuracy(logits, tgt_output_ids, vocab.PAD_ID)

            loss_sum += loss.item()
            acc_sum += acc
            num_batches += 1

    avg_loss = loss_sum / num_batches if num_batches > 0 else 0.0
    avg_acc = acc_sum / num_batches if num_batches > 0 else 0.0

    exact_no_ttt = exact_match_accuracy(
        model=model,
        dataloader=loader,
        vocab=vocab,
        representation=representation,
        device=device,
        max_decode_len=max_decode_len,
    )

    print("\nModel without TTT")
    print(f"Loss: {avg_loss:.6f}")
    print(f"Token accuracy with teacher forcing: {avg_acc:.6f}")
    print(f"Exact match without teacher forcing: {exact_no_ttt:.6f}")


    print("\nModel with TTT")

    base_ttt_state = copy.deepcopy(model.ttt.state_dict())

    loss_sum_ttt = 0.0
    acc_sum_ttt = 0.0
    num_examples_ttt = 0

    for batch in loader:
        a_list = batch["a"]
        b_list = batch["b"]
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        tgt_output_ids = batch["tgt_output_ids"].to(device)

        for i in range(len(a_list)):
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
                inner_function= inner_loss
            )

            one_tgt_in = tgt_input_ids[i:i+1]
            one_tgt_out = tgt_output_ids[i:i+1]

            with torch.no_grad():
                logits, _ = model.decoder(one_tgt_in, adapted_hidden)

                loss = criterion(
                    logits.reshape(-1, vocab.VOCAB_SIZE),
                    one_tgt_out.reshape(-1)
                )

                acc = token_accuracy(logits, one_tgt_out, vocab.PAD_ID)

            loss_sum_ttt += loss.item()
            acc_sum_ttt += acc
            num_examples_ttt += 1

    model.ttt.load_state_dict(base_ttt_state)

    avg_loss_ttt = loss_sum_ttt / num_examples_ttt if num_examples_ttt > 0 else 0.0
    avg_acc_ttt = acc_sum_ttt / num_examples_ttt if num_examples_ttt > 0 else 0.0

    exact_ttt = exact_match_accuracy_with_ttt(
        model=model,
        dataloader=loader,
        vocab=vocab,
        representation=representation,
        device=device,
        max_decode_len=max_decode_len,
        ttt_steps=ttt_steps,
        ttt_lr=ttt_lr,
    )

    print(f"Loss: {avg_loss_ttt:.6f}")
    print(f"Token accuracy with teacher forcing: {avg_acc_ttt:.6f}")
    print(f"Exact match with TTT: {exact_ttt:.6f}")

    print("\nModel with TTT online")

    model.ttt.load_state_dict(base_ttt_state)

    loss_sum_ttt_online = 0.0
    acc_sum_ttt_online = 0.0
    num_examples_ttt_online = 0

    for batch in loader:
        a_list = batch["a"]
        b_list = batch["b"]
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        tgt_output_ids = batch["tgt_output_ids"].to(device)

        for i in range(len(a_list)):
            # no reset here: this is online TTT
            adapted_hidden = adapt_ttt_for_one_example(
                model=model,
                a=a_list[i],
                b=b_list[i],
                vocab=vocab,
                representation=representation,
                device=device,
                ttt_steps=ttt_steps,
                ttt_lr=ttt_lr,
                inner_function=inner_loss
            )

            one_tgt_in = tgt_input_ids[i:i+1]
            one_tgt_out = tgt_output_ids[i:i+1]

            with torch.no_grad():
                logits, _ = model.decoder(one_tgt_in, adapted_hidden)

                loss = criterion(
                    logits.reshape(-1, vocab.VOCAB_SIZE),
                    one_tgt_out.reshape(-1)
                )

                acc = token_accuracy(logits, one_tgt_out, vocab.PAD_ID)

            loss_sum_ttt_online += loss.item()
            acc_sum_ttt_online += acc
            num_examples_ttt_online += 1

    model.ttt.load_state_dict(base_ttt_state)

    avg_loss_ttt_online = (
        loss_sum_ttt_online / num_examples_ttt_online
        if num_examples_ttt_online > 0 else 0.0
    )
    avg_acc_ttt_online = (
        acc_sum_ttt_online / num_examples_ttt_online
        if num_examples_ttt_online > 0 else 0.0
    )

    exact_ttt_online = exact_match_accuracy_with_ttt_online(
        model=model,
        dataloader=loader,
        vocab=vocab,
        representation=representation,
        device=device,
        max_decode_len=max_decode_len,
        ttt_steps=ttt_steps,
        ttt_lr=ttt_lr,
    )

    model.ttt.load_state_dict(base_ttt_state)

    print(f"Loss: {avg_loss_ttt_online:.6f}")
    print(f"Token accuracy with teacher forcing: {avg_acc_ttt_online:.6f}")
    print(f"Exact match with TTT online: {exact_ttt_online:.6f}")
    print()

    return {
        "loss": avg_loss,
        "token_accuracy": avg_acc,
        "exact_match": exact_no_ttt,

        "loss_ttt": avg_loss_ttt,
        "token_accuracy_ttt": avg_acc_ttt,
        "exact_match_ttt": exact_ttt,

        "loss_ttt_online": avg_loss_ttt_online,
        "token_accuracy_ttt_online": avg_acc_ttt_online,
        "exact_match_ttt_online": exact_ttt_online,
    }