from src.data.collate_seq2seq import *
from src.evaluation.metrics_seq2seq import *
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def train_model(
    representation="binary",
    num_digits=3,
    train_samples=20000,
    valid_samples=2000,
    batch_size=128,
    emb_dim=32,
    hidden_dim=128,
    ttt_bottleneck_dim=None,
    epochs=500,
    lr=1e-3,
    patience=10,
    print_every=1,
    device=None,
    load_checkpoint=False,
    checkpoint_path="outputs/seq2seq_with_gru.pt"
):
    vocab = build_vocab(representation)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Representation: {representation}")
    print(f"Vocab tokens: {vocab.tokens}")

    train_dataset = AdditionDataset(
        vocab=vocab,
        num_samples=train_samples,
        num_digits=num_digits,
        representation=representation
    )
    valid_dataset = AdditionDataset(
        vocab=vocab,
        num_samples=valid_samples,
        num_digits=num_digits,
        representation=representation
    )

    collate_fn = make_collate_fn(vocab)

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

    model = Seq2SeqGRUWithTTT(
        vocab=vocab,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        ttt_bottleneck_dim=ttt_bottleneck_dim,
    ).to(device)

    if load_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

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
        train_acc_sum = 0.0
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

            optimizer.zero_grad()

            # TRAIN: ordinary forward without TTT adaptation
            logits = model(src_ids, src_lens, tgt_input_ids,current_p= p_current, scheduled_sampling=True)

            loss = criterion(
                logits.reshape(-1, vocab.VOCAB_SIZE),
                tgt_output_ids.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            acc = token_accuracy(logits, tgt_output_ids, vocab.PAD_ID)

            train_loss_sum += loss.item()
            train_acc_sum += acc
            train_steps += 1

        train_loss = train_loss_sum / train_steps
        train_acc = train_acc_sum / train_steps

        model.eval()
        valid_loss_sum = 0.0
        valid_acc_sum = 0.0
        valid_steps = 0

        with torch.no_grad():
            for batch in valid_loader:
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

                valid_loss_sum += loss.item()
                valid_acc_sum += acc
                valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        valid_acc = valid_acc_sum / valid_steps

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
            torch.save(model.state_dict(), "outputs/seq2seq_with_gru.pt")
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
                f"train_loss={train_loss:.4f} | train_token_acc={train_acc:.4f} | "
                f"valid_loss={valid_loss:.4f} | valid_token_acc={valid_acc:.4f} | "
                f"valid_exact_match={valid_exact:.4f}"
            )

    model.load_state_dict(torch.load("outputs/seq2seq_with_gru.pt", map_location=device))
    return model, vocab, train_dataset, valid_dataset, valid_loader, device