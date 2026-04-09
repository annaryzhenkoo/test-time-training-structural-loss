from src.data.collate_seq2seq import *
from src.evaluation.metrics_seq2seq import *
from src.data.datasets_seq2seq import *
from src.models.seq2seq_gru import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def train_model_with_carry(
    representation="binary",
    num_digits=3,
    train_samples=20000,
    valid_samples=2000,
    batch_size=128,
    emb_dim=32,
    hidden_dim=128,
    ttt_bottleneck_dim=None,
    carry_emb_dim=4,
    epochs=500,
    lr=1e-3,
    patience=10,
    print_every=1,
    device=None,
    checkpoint_path="outputs/seq2seq_with_gru_and_carry.pt",
    carry_loss_weight=1.0,
):
    vocab = build_vocab(representation)
    carry_vocab = build_carry_vocab()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Representation: {representation}")
    print(f"Vocab tokens: {vocab.tokens}")
    print(f"Carry vocab tokens: {carry_vocab.tokens}")

    train_dataset = AdditionDatasetWithCarry(
        vocab=vocab,
        carry_vocab=carry_vocab,
        num_samples=train_samples,
        num_digits=num_digits,
        representation=representation
    )

    valid_dataset = AdditionDatasetWithCarry(
        vocab=vocab,
        carry_vocab=carry_vocab,
        num_samples=valid_samples,
        num_digits=num_digits,
        representation=representation
    )

    collate_fn = make_collate_fn_with_carry(vocab, carry_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Seq2SeqGRUWithTTTandCarry(
        vocab=vocab,
        carry_vocab=carry_vocab,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        ttt_bottleneck_dim=ttt_bottleneck_dim,
        carry_emb_dim=carry_emb_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    token_criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    carry_criterion = nn.CrossEntropyLoss(ignore_index=carry_vocab.PAD_ID)

    best_valid_loss = np.inf
    patience_counter = 0

    if num_digits == 3:
        max_value = 999
    else:
        max_value = 10 ** num_digits - 1

    if representation == "binary":
        max_decode_len = len(bin(max_value * 2)[2:]) + 2
    else:
        max_decode_len = len(str(max_value * 2)) + 2

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_token_loss_sum = 0.0
        train_carry_loss_sum = 0.0
        train_token_acc_sum = 0.0
        train_carry_acc_sum = 0.0
        train_steps = 0

        subsample_epochs = min(100, epochs)
        pmin = 0
        if epoch > 1:
            p_current = max(pmin, 1 - epoch / subsample_epochs)
        else:
            p_current = 1.0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            src_lens = batch["src_lens"].to(device)

            tgt_input_ids = batch["tgt_input_ids"].to(device)
            tgt_output_ids = batch["tgt_output_ids"].to(device)

            carry_input_ids = batch["carry_input_ids"].to(device)
            carry_output_ids = batch["carry_output_ids"].to(device)

            optimizer.zero_grad()

            token_logits, carry_logits = model.forward_with_carry(
                src_ids=src_ids,
                src_lens=src_lens,
                tgt_input_ids=tgt_input_ids,
                carry_input_ids=carry_input_ids,
                current_p=p_current,
            )

            token_loss = token_criterion(
                token_logits.reshape(-1, vocab.VOCAB_SIZE),
                tgt_output_ids.reshape(-1)
            )

            carry_loss = carry_criterion(
                carry_logits.reshape(-1, carry_vocab.VOCAB_SIZE),
                carry_output_ids.reshape(-1)
            )

            loss = token_loss + carry_loss_weight * carry_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            token_acc = token_accuracy(token_logits, tgt_output_ids, vocab.PAD_ID)
            carry_acc = token_accuracy(carry_logits, carry_output_ids, carry_vocab.PAD_ID)

            train_loss_sum += loss.item()
            train_token_loss_sum += token_loss.item()
            train_carry_loss_sum += carry_loss.item()
            train_token_acc_sum += token_acc
            train_carry_acc_sum += carry_acc
            train_steps += 1

        train_loss = train_loss_sum / train_steps
        train_token_loss = train_token_loss_sum / train_steps
        train_carry_loss = train_carry_loss_sum / train_steps
        train_token_acc = train_token_acc_sum / train_steps
        train_carry_acc = train_carry_acc_sum / train_steps

        model.eval()
        valid_loss_sum = 0.0
        valid_token_loss_sum = 0.0
        valid_carry_loss_sum = 0.0
        valid_token_acc_sum = 0.0
        valid_carry_acc_sum = 0.0
        valid_steps = 0

        with torch.no_grad():
            for batch in valid_loader:
                src_ids = batch["src_ids"].to(device)
                src_lens = batch["src_lens"].to(device)

                tgt_input_ids = batch["tgt_input_ids"].to(device)
                tgt_output_ids = batch["tgt_output_ids"].to(device)

                carry_input_ids = batch["carry_input_ids"].to(device)
                carry_output_ids = batch["carry_output_ids"].to(device)

                token_logits, carry_logits = model.forward_with_carry(
                    src_ids=src_ids,
                    src_lens=src_lens,
                    tgt_input_ids=tgt_input_ids,
                    carry_input_ids=carry_input_ids,
                    current_p=1.0,
                )

                token_loss = token_criterion(
                    token_logits.reshape(-1, vocab.VOCAB_SIZE),
                    tgt_output_ids.reshape(-1)
                )

                carry_loss = carry_criterion(
                    carry_logits.reshape(-1, carry_vocab.VOCAB_SIZE),
                    carry_output_ids.reshape(-1)
                )

                loss = token_loss + carry_loss_weight * carry_loss

                token_acc = token_accuracy(token_logits, tgt_output_ids, vocab.PAD_ID)
                carry_acc = token_accuracy(carry_logits, carry_output_ids, carry_vocab.PAD_ID)

                valid_loss_sum += loss.item()
                valid_token_loss_sum += token_loss.item()
                valid_carry_loss_sum += carry_loss.item()
                valid_token_acc_sum += token_acc
                valid_carry_acc_sum += carry_acc
                valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        valid_token_loss = valid_token_loss_sum / valid_steps
        valid_carry_loss = valid_carry_loss_sum / valid_steps
        valid_token_acc = valid_token_acc_sum / valid_steps
        valid_carry_acc = valid_carry_acc_sum / valid_steps

        valid_exact = exact_match_accuracy(
            model,
            valid_loader,
            vocab=vocab,
            representation=representation,
            device=device,
            max_decode_len=max_decode_len
        )

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best valid loss = {best_valid_loss:.4f}")
                break

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"p_current={p_current:.3f} | "
                f"train_loss={train_loss:.4f} | "
                f"train_token_loss={train_token_loss:.4f} | "
                f"train_carry_loss={train_carry_loss:.4f} | "
                f"train_token_acc={train_token_acc:.4f} | "
                f"train_carry_acc={train_carry_acc:.4f} | "
                f"valid_loss={valid_loss:.4f} | "
                f"valid_token_loss={valid_token_loss:.4f} | "
                f"valid_carry_loss={valid_carry_loss:.4f} | "
                f"valid_token_acc={valid_token_acc:.4f} | "
                f"valid_carry_acc={valid_carry_acc:.4f} | "
                f"valid_exact_match={valid_exact:.4f}"
            )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, vocab, carry_vocab, train_dataset, valid_dataset, valid_loader, device