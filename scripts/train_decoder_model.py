import math
import os
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data.vocab_decoder import Vocab, build_vocab
from src.helpers import *
from src.data.decoder_data import *
from  src.models.decoder_model import *



# =========================================================
# 6. TRAIN / EVAL
# =========================================================

def token_accuracy(logits, target_ids, pad_id: int):
    preds = logits.argmax(dim=-1)
    mask = (target_ids != pad_id)

    if mask.sum().item() == 0:
        return 0.0

    correct = ((preds == target_ids) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


@torch.no_grad()
def greedy_generate(
        model,
        a: int,
        b: int,
        vocab: Vocab,
        representation: str,
        device: str = "cpu",
        max_new_tokens: int = 32,
        offset: int = 0,
):
    model.eval()

    a_str = int_to_base_string(a, representation)
    b_str = int_to_base_string(b, representation)
    L = max(len(a_str), len(b_str))

    a_pad = a_str.zfill(L)
    b_pad = b_str.zfill(L)

    prefix_tokens = list(reverse_str(a_pad)) + ["+"] + list(reverse_str(b_pad)) + ["="]
    prefix_ids = [vocab.stoi[t] for t in prefix_tokens]

    prefix_pos_ids = build_position_ids_for_input(L=L, offset=offset)[: len(prefix_tokens)]

    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(prefix_pos_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_answer_len = 0

    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids, position_ids=position_ids)
        next_id = logits[:, -1, :].argmax(dim=-1)
        next_token_id = next_id.item()

        input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

        generated_answer_len += 1

        if next_token_id == vocab.EOS_ID:
            next_pos = 0
        else:
            next_pos = offset + generated_answer_len

        prefix_pos_ids.append(next_pos)
        position_ids = torch.tensor(
            prefix_pos_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        if next_token_id == vocab.EOS_ID:
            break

    out_ids = input_ids[0].tolist()
    out_tokens = decode_ids(out_ids, vocab)
    pred_int = parse_prediction_to_int(out_tokens, representation=representation)
    return out_tokens, pred_int


@torch.no_grad()
def exact_match_accuracy(
        model,
        dataloader,
        vocab: Vocab,
        representation: str,
        device: str = "cpu",
):
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        a_list = batch["a"]
        b_list = batch["b"]
        c_list = batch["c"]

        for a, b, c_true in zip(a_list, b_list, c_list):
            L = max(
                len(int_to_base_string(a, representation)),
                len(int_to_base_string(b, representation)),
            )
            _, pred = greedy_generate(
                model=model,
                a=a,
                b=b,
                vocab=vocab,
                representation=representation,
                device=device,
                max_new_tokens=L + 3,
                offset=0,
            )
            if pred == c_true:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_on_range(
        model,
        vocab: Vocab,
        representation: str,
        min_value: int,
        max_value: int,
        num_samples: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 999,
):
    eval_dataset = AdditionDataset(
        vocab=vocab,
        num_samples=num_samples,
        min_value=min_value,
        max_value=max_value,
        representation=representation,
        random_shift_positions=False,
        max_pos_offset=0,
        seed=seed,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(vocab.PAD_ID),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

    model.eval()

    loss_sum = 0.0
    tok_acc_sum = 0.0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        position_ids = batch["position_ids"].to(device)

        logits = model(input_ids=input_ids, position_ids=position_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        loss_sum += loss.item()
        tok_acc_sum += token_accuracy(logits, target_ids, vocab.PAD_ID)

    avg_loss = loss_sum / len(eval_loader)
    avg_tok_acc = tok_acc_sum / len(eval_loader)

    em = exact_match_accuracy(
        model=model,
        dataloader=eval_loader,
        vocab=vocab,
        representation=representation,
        device=device,
    )

    return {
        "loss": avg_loss,
        "token_accuracy": avg_tok_acc,
        "exact_match": em,
    }


def train_model(
        model,
        train_loader,
        valid_loader,
        vocab: Vocab,
        representation: str,
        device: str = "cpu",
        epochs: int = 30,
        lr: float = 3e-4,
        ood_eval_samples: int = 1000,
        save_path: str = "best_model.pt",
        print_every: int = 5,
):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

    best_valid_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            position_ids = batch["position_ids"].to(device)

            logits = model(input_ids=input_ids, position_ids=position_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += token_accuracy(logits, target_ids, vocab.PAD_ID)

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_acc = train_acc_sum / len(train_loader)

        model.eval()
        valid_loss_sum = 0.0
        valid_acc_sum = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                position_ids = batch["position_ids"].to(device)

                logits = model(input_ids=input_ids, position_ids=position_ids)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )

                valid_loss_sum += loss.item()
                valid_acc_sum += token_accuracy(logits, target_ids, vocab.PAD_ID)

        avg_valid_loss = valid_loss_sum / len(valid_loader)
        avg_valid_acc = valid_acc_sum / len(valid_loader)

        improved = avg_valid_loss < best_valid_loss
        if improved:
            best_valid_loss = avg_valid_loss

            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    "representation": representation,
                    "vocab_tokens": vocab.tokens,
                    "model_config": {
                        "vocab_size": model.vocab_size,
                        "pad_id": model.pad_id,
                        "d_model": model.d_model,
                    },
                },
                save_path,
            )

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            valid_em = exact_match_accuracy(
                model=model,
                dataloader=valid_loader,
                vocab=vocab,
                representation=representation,
                device=device,
            )

            ood_metrics = evaluate_on_range(
                model=model,
                vocab=vocab,
                representation=representation,
                min_value=1000,
                max_value=9999,
                num_samples=ood_eval_samples,
                batch_size=valid_loader.batch_size,
                device=device,
                seed=1000 + epoch,
            )


            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"train_tok_acc={avg_train_acc:.4f} | "
                f"valid_loss={avg_valid_loss:.4f} | "
                f"valid_tok_acc={avg_valid_acc:.4f} | "
                f"valid_EM={valid_em:.4f} | "
                f"OOD1000_9999_loss={ood_metrics['loss']:.4f} | "
                f"OOD1000_9999_tok_acc={ood_metrics['token_accuracy']:.4f} | "
                f"OOD1000_9999_EM={ood_metrics['exact_match']:.4f}"
                + (" | saved_best_model" if improved else "")
            )

    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Best model saved to: {save_path}")


# =========================================================
# 7. RUN
# =========================================================

def build_loader(
        vocab: Vocab,
        num_samples: int,
        min_value: int,
        max_value: int,
        batch_size: int,
        shuffle: bool,
        random_shift_positions: bool,
        max_pos_offset: int,
        seed: int,
        representation: str,
):
    dataset = AdditionDataset(
        vocab=vocab,
        num_samples=num_samples,
        min_value=min_value,
        max_value=max_value,
        representation=representation,
        random_shift_positions=random_shift_positions,
        max_pos_offset=max_pos_offset,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(vocab.PAD_ID),
    )
    return loader


def estimate_max_binary_len(max_value: int) -> int:
    return len(bin(max_value)[2:])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    representation = "binary"
    print("representation:", representation)

    # Train on decimal range 0..999
    train_min_value = 0
    train_max_value = 999

    # OOD evaluate on decimal range 1000..9999
    ood_min_value = 1000
    ood_max_value = 9999

    train_samples = 20000
    valid_samples = 2000
    ood_eval_samples = 1000

    batch_size = 128
    epochs = 200
    lr = 1e-3

    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.1

    max_pos_offset = 64

    vocab = build_vocab(representation)

    train_loader = build_loader(
        vocab=vocab,
        num_samples=train_samples,
        min_value=train_min_value,
        max_value=train_max_value,
        batch_size=batch_size,
        shuffle=True,
        random_shift_positions=True,
        max_pos_offset=max_pos_offset,
        seed=42,
        representation=representation,
    )

    valid_loader = build_loader(
        vocab=vocab,
        num_samples=valid_samples,
        min_value=train_min_value,
        max_value=train_max_value,
        batch_size=batch_size,
        shuffle=False,
        random_shift_positions=False,
        max_pos_offset=max_pos_offset,
        seed=123,
        representation=representation,
    )

    # Need position space large enough for OOD binary lengths too
    max_ood_operand_len = estimate_max_binary_len(ood_max_value)  # for 9999
    max_position_id = max_pos_offset + (max_ood_operand_len + 1)

    print("max_ood_operand_len:", max_ood_operand_len)
    print("max_position_id:", max_position_id)

    model = DecoderOnlyAdditionTransformer(
        vocab_size=vocab.VOCAB_SIZE,
        pad_id=vocab.PAD_ID,
        max_position_id=max_position_id,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab=vocab,
        representation=representation,
        device=device,
        epochs=epochs,
        lr=lr,
        ood_eval_samples=ood_eval_samples,
        save_path="/kaggle/working/best_model.pt",
        print_every=5,
    )

    print("\n=== seen-range checks: 0..999 ===")
    seen_cases = [
        (34, 5),
        (507, 92),
        (999, 999),
        (120, 305),
    ]

    for a, b in seen_cases:
        tokens, pred = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max(
                len(int_to_base_string(a, representation)),
                len(int_to_base_string(b, representation)),
            ) + 3,
            offset=6,
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"a_bin={int_to_base_string(a, representation)} | "
            f"b_bin={int_to_base_string(b, representation)} | "
            f"pred={pred} | tokens={tokens}"
        )

    print("\n=== seen-range checks: 1000..9999 ===")
    seen_cases = [
        (1001, 5),
        (2031, 1044),
        (9999, 9999),
        (1230, 3051),
    ]

    for a, b in seen_cases:
        tokens, pred = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max(
                len(int_to_base_string(a, representation)),
                len(int_to_base_string(b, representation)),
            ) + 3,
            offset=6,
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"a_bin={int_to_base_string(a, representation)} | "
            f"b_bin={int_to_base_string(b, representation)} | "
            f"pred={pred} | tokens={tokens}"
        )

    print("\n=== OOD-range evaluation: 1000..9999 ===")
    ood_metrics = evaluate_on_range(
        model=model,
        vocab=vocab,
        representation=representation,
        min_value=ood_min_value,
        max_value=ood_max_value,
        num_samples=2000,
        batch_size=batch_size,
        device=device,
        seed=777,
    )
    print(
        f"OOD[1000,9999] | "
        f"loss={ood_metrics['loss']:.4f} | "
        f"tok_acc={ood_metrics['token_accuracy']:.4f} | "
        f"EM={ood_metrics['exact_match']:.4f}"
    )

    print("\n=== OOD checks: decimal numbers 1000..9999, but model input is binary ===")
    ood_cases = [
        (1000, 1),
        (1234, 5678),
        (9999, 1),
        (4321, 876),
    ]

    for a, b in ood_cases:
        tokens, pred = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max(
                len(int_to_base_string(a, representation)),
                len(int_to_base_string(b, representation)),
            ) + 3,
            offset=6,
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"a_bin={int_to_base_string(a, representation)} | "
            f"b_bin={int_to_base_string(b, representation)} | "
            f"pred={pred} | tokens={tokens}"
        )


if __name__ == "__main__":
    main()