
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class Vocab:
    tokens: list
    stoi: dict
    itos: dict
    PAD_ID: int
    EOS_ID: int
    VOCAB_SIZE: int


def build_vocab(representation: str) -> Vocab:
    if representation == "decimal":
        digit_tokens = [str(i) for i in range(10)]
    elif representation == "binary":
        digit_tokens = ["0", "1"]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    tokens = ["<PAD>", "<EOS>"] + digit_tokens + ["+", "="]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for i, tok in enumerate(tokens)}

    return Vocab(
        tokens=tokens,
        stoi=stoi,
        itos=itos,
        PAD_ID=stoi["<PAD>"],
        EOS_ID=stoi["<EOS>"],
        VOCAB_SIZE=len(tokens),
    )


def int_to_base_string(n: int, representation: str) -> str:
    if representation == "decimal":
        return str(n)
    elif representation == "binary":
        return bin(n)[2:]
    else:
        raise ValueError(f"Unknown representation: {representation}")


def to_padded_base_string(n: int, total_len: int, representation: str) -> str:
    return int_to_base_string(n, representation).zfill(total_len)


def reverse_str(s: str) -> str:
    return s[::-1]


def parse_digit_string_to_int(s: str, representation: str):
    try:
        if representation == "decimal":
            return int(s, 10)
        elif representation == "binary":
            return int(s, 2)
        else:
            raise ValueError(f"Unknown representation: {representation}")
    except ValueError:
        return None


def encode_tokens(tokens, vocab: Vocab):
    return [vocab.stoi[t] for t in tokens]


def decode_ids(ids, vocab: Vocab):
    return [vocab.itos[i] for i in ids]


def build_addition_sequence(a: int, b: int, representation: str):
    c = a + b

    a_str = int_to_base_string(a, representation)
    b_str = int_to_base_string(b, representation)

    L = max(len(a_str), len(b_str))

    a_pad = a_str.zfill(L)
    b_pad = b_str.zfill(L)
    c_pad = int_to_base_string(c, representation).zfill(L + 1)

    a_rev = reverse_str(a_pad)
    b_rev = reverse_str(b_pad)
    c_rev = reverse_str(c_pad)

    tokens = list(a_rev) + ["+"] + list(b_rev) + ["="] + list(c_rev) + ["<EOS>"]

    return {
        "a": a,
        "b": b,
        "c": c,
        "L": L,
        "a_str": a_str,
        "b_str": b_str,
        "c_str": int_to_base_string(c, representation),
        "a_pad": a_pad,
        "b_pad": b_pad,
        "c_pad": c_pad,
        "tokens": tokens,
    }


def parse_prediction_to_int(tokens, representation: str):
    if "=" not in tokens:
        return None

    eq_pos = tokens.index("=")
    answer_tokens = []

    for t in tokens[eq_pos + 1:]:
        if t == "<EOS>":
            break
        answer_tokens.append(t)

    if len(answer_tokens) == 0:
        return None

    normal_order = "".join(answer_tokens[::-1])
    return parse_digit_string_to_int(normal_order, representation)



def build_position_ids_for_input(L: int, offset: int = 0):
    digit_ids = [offset + i for i in range(1, L + 1)]
    sum_digit_ids = [offset + i for i in range(1, L + 2)]

    sep_id = offset + L + 2

    pos_ids = []
    pos_ids += digit_ids
    pos_ids += [sep_id]
    pos_ids += digit_ids
    pos_ids += [sep_id]
    pos_ids += sum_digit_ids

    return pos_ids



class AdditionDataset(Dataset):
    def __init__(
        self,
        vocab: Vocab,
        num_samples: int,
        min_value: int,
        max_value: int,
        representation: str = "binary",
        random_shift_positions: bool = True,
        max_pos_offset: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab = vocab
        self.num_samples = num_samples
        self.min_value = min_value
        self.max_value = max_value
        self.representation = representation
        self.random_shift_positions = random_shift_positions
        self.max_pos_offset = max_pos_offset

        self.rng = random.Random(seed)
        self.examples = []

        for _ in range(num_samples):
            a = self.rng.randint(min_value, max_value)
            b = self.rng.randint(min_value, max_value)
            self.examples.append((a, b))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, b = self.examples[idx]

        ex = build_addition_sequence(
            a=a,
            b=b,
            representation=self.representation,
        )

        L = ex["L"]

        a_rev_tokens = list(reverse_str(ex["a_pad"]))
        b_rev_tokens = list(reverse_str(ex["b_pad"]))
        c_rev_tokens = list(reverse_str(ex["c_pad"]))

        prefix_tokens = a_rev_tokens + ["+"] + b_rev_tokens + ["="]
        answer_tokens = c_rev_tokens
        eos_token = ["<EOS>"]

        input_tokens = prefix_tokens + answer_tokens
        prefix_len = len(prefix_tokens)

        target_tokens = (
            ["<PAD>"] * (prefix_len - 1)
            + [answer_tokens[0]]
            + answer_tokens[1:]
            + eos_token
        )

        assert len(input_tokens) == len(target_tokens), (
            f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}"
        )

        input_ids = encode_tokens(input_tokens, self.vocab)
        target_ids = encode_tokens(target_tokens, self.vocab)

        if self.random_shift_positions:
            offset = self.rng.randint(0, self.max_pos_offset)
        else:
            offset = 0

        input_pos_ids = build_position_ids_for_input(L=L, offset=offset)

        assert len(input_ids) == len(input_pos_ids), (
            f"Length mismatch: input_ids={len(input_ids)}, pos_ids={len(input_pos_ids)}"
        )

        return {
            "a": a,
            "b": b,
            "c": a + b,
            "L": L,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "position_ids": torch.tensor(input_pos_ids, dtype=torch.long),
            "text_tokens": ex["tokens"],
            "a_str": ex["a_str"],
            "b_str": ex["b_str"],
            "c_str": ex["c_str"],
        }


def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)

        input_batch = []
        target_batch = []
        pos_batch = []

        for item in batch:
            inp = item["input_ids"]
            tgt = item["target_ids"]
            pos = item["position_ids"]

            inp_pad = torch.full((max_len,), pad_id, dtype=torch.long)
            tgt_pad = torch.full((max_len,), pad_id, dtype=torch.long)
            pos_pad = torch.zeros((max_len,), dtype=torch.long)

            inp_pad[: inp.size(0)] = inp
            tgt_pad[: tgt.size(0)] = tgt
            pos_pad[: pos.size(0)] = pos

            input_batch.append(inp_pad)
            target_batch.append(tgt_pad)
            pos_batch.append(pos_pad)

        return {
            "input_ids": torch.stack(input_batch, dim=0),
            "target_ids": torch.stack(target_batch, dim=0),
            "position_ids": torch.stack(pos_batch, dim=0),
            "a": [x["a"] for x in batch],
            "b": [x["b"] for x in batch],
            "c": [x["c"] for x in batch],
            "L": [x["L"] for x in batch],
            "a_str": [x["a_str"] for x in batch],
            "b_str": [x["b_str"] for x in batch],
            "c_str": [x["c_str"] for x in batch],
        }

    return collate_fn



@dataclass
class TTTConfig:
    hidden_dim: int = 256
    dropout: float = 0.0


class TTTAdapter(nn.Module):
    def __init__(self, d_model: int, cfg: TTTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, d_model),
        )

    def forward(self, h: torch.Tensor):
        return h + self.net(h)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(
        self,
        max_position_id: int,
        d_model: int,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.max_position_id = max_position_id
        self.d_model = d_model
        self.padding_idx = padding_idx

        pe = torch.zeros(max_position_id + 1, d_model)

        position = torch.arange(
            0,
            max_position_id + 1,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        # Works for both even and odd d_model.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        pe[padding_idx] = 0.0

        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor):
        max_used_position_id = int(position_ids.max().item())

        if max_used_position_id > self.max_position_id:
            raise ValueError(
                f"position_ids contain value {max_used_position_id}, "
                f"but max_position_id={self.max_position_id}. "
                f"Increase max_position_id in main()."
            )

        return self.pe[position_ids]


class DecoderOnlyAdditionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_position_id: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        ttt_hidden_dim: int = 256,
        ttt_dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_position_id = max_position_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.position_embedding = SinusoidalPositionEmbedding(
            max_position_id=max_position_id,
            d_model=d_model,
            padding_idx=0,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.ttt_adapter = TTTAdapter(
            d_model=d_model,
            cfg=TTTConfig(hidden_dim=ttt_hidden_dim, dropout=ttt_dropout),
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode_hidden_before_adapter(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        pos_emb = self.position_embedding(position_ids)

        x = token_emb + pos_emb

        causal_mask = self.make_causal_mask(T, device=device)
        padding_mask = (input_ids == self.pad_id)

        h = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        return_hidden: bool = False,
    ):
        h = self.encode_hidden_before_adapter(input_ids, position_ids)
        z = self.ttt_adapter(h)
        logits = self.output_proj(z)

        if return_hidden:
            return logits, h, z
        return logits


def clone_ttt_adapter_params(model: DecoderOnlyAdditionTransformer):
    return {
        k: v.detach().clone()
        for k, v in model.ttt_adapter.state_dict().items()
    }


def restore_ttt_adapter_params(model: DecoderOnlyAdditionTransformer, state_dict: dict):
    model.ttt_adapter.load_state_dict(state_dict, strict=True)


def save_requires_grad_state(model: nn.Module):
    return {name: p.requires_grad for name, p in model.named_parameters()}


def restore_requires_grad_state(model: nn.Module, requires_grad_state: dict):
    for name, p in model.named_parameters():
        p.requires_grad = requires_grad_state[name]


def freeze_except_ttt_adapter_temporarily(model: DecoderOnlyAdditionTransformer):
    requires_grad_state = save_requires_grad_state(model)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.ttt_adapter.parameters():
        p.requires_grad = True

    return requires_grad_state


def build_prefix_input(
    a: int,
    b: int,
    vocab: Vocab,
    representation: str,
    device: str = "cpu",
    offset: int = 0,
):
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

    return input_ids, position_ids, L


def ttt_adapt_on_example(
    model: DecoderOnlyAdditionTransformer,
    a: int,
    b: int,
    vocab: Vocab,
    representation: str,
    device: str = "cpu",
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
    l2_lambda: float = 1e-3,
    offset: int = 0,
    loss_type: str = "mse",  # mse | smooth_l1 | cosine
):
    original_requires_grad = freeze_except_ttt_adapter_temporarily(model)
    previous_mode_is_training = model.training
    model.train()

    init_state = clone_ttt_adapter_params(model)
    optimizer = torch.optim.SGD(model.ttt_adapter.parameters(), lr=ttt_lr)

    x_ab, pos_ab, _ = build_prefix_input(
        a=a,
        b=b,
        vocab=vocab,
        representation=representation,
        device=device,
        offset=offset,
    )
    x_ba, pos_ba, _ = build_prefix_input(
        a=b,
        b=a,
        vocab=vocab,
        representation=representation,
        device=device,
        offset=offset,
    )

    info = {
        "final_loss": None,
        "final_comm_loss": None,
        "final_l2": None,
    }

    try:
        for _ in range(ttt_steps):
            optimizer.zero_grad()

            _, _, z_ab = model(x_ab, pos_ab, return_hidden=True)
            _, _, z_ba = model(x_ba, pos_ba, return_hidden=True)

            z_ab_eq = z_ab[:, -1, :]
            z_ba_eq = z_ba[:, -1, :]

            if loss_type == "mse":
                comm_loss = F.mse_loss(z_ab_eq, z_ba_eq)
            elif loss_type == "smooth_l1":
                comm_loss = F.smooth_l1_loss(z_ab_eq, z_ba_eq)
            elif loss_type == "cosine":
                cos_sim = F.cosine_similarity(z_ab_eq, z_ba_eq, dim=-1)
                comm_loss = (1.0 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            l2 = torch.zeros((), device=device)
            for name, p in model.ttt_adapter.named_parameters():
                p0 = init_state[name].to(p.device)
                l2 = l2 + (p - p0).pow(2).mean()

            loss = comm_loss + l2_lambda * l2
            loss.backward()
            optimizer.step()

            info["final_loss"] = float(loss.item())
            info["final_comm_loss"] = float(comm_loss.item())
            info["final_l2"] = float(l2.item())

    finally:
        restore_requires_grad_state(model, original_requires_grad)
        if previous_mode_is_training:
            model.train()
        else:
            model.eval()

    return init_state, info



def token_accuracy(logits, target_ids, pad_id: int):
    preds = logits.argmax(dim=-1)
    mask = (target_ids != pad_id)

    if mask.sum().item() == 0:
        return 0.0

    correct = ((preds == target_ids) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


def teacher_forcing_probability(epoch: int, decay_until_epoch: int = 75):
    if decay_until_epoch <= 1:
        return 0.0

    if epoch >= decay_until_epoch:
        return 0.0

    return 1.0 - (epoch - 1) / (decay_until_epoch - 1)


def forward_with_scheduled_sampling(
    model: DecoderOnlyAdditionTransformer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    position_ids: torch.Tensor,
    pad_id: int,
    teacher_forcing_prob: float,
):
    current_input_ids = input_ids.clone()
    B, T = input_ids.shape

    for t in range(T - 1):
        logits = model(
            input_ids=current_input_ids,
            position_ids=position_ids,
        )

        gold_next = target_ids[:, t]
        pred_next = logits[:, t, :].argmax(dim=-1).detach()

        active = gold_next != pad_id
        if not active.any():
            continue

        use_teacher = torch.rand(B, device=input_ids.device) < teacher_forcing_prob

        next_token = torch.where(
            use_teacher,
            gold_next,
            pred_next,
        )

        current_input_ids[active, t + 1] = next_token[active]

    logits = model(
        input_ids=current_input_ids,
        position_ids=position_ids,
    )

    return logits


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

    input_ids, position_ids, _ = build_prefix_input(
        a=a,
        b=b,
        vocab=vocab,
        representation=representation,
        device=device,
        offset=offset,
    )

    generated_answer_len = 0
    prefix_pos_ids = position_ids[0].tolist()

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


def greedy_generate_with_ttt(
    model,
    a: int,
    b: int,
    vocab: Vocab,
    representation: str,
    device: str = "cpu",
    max_new_tokens: int = 32,
    offset: int = 0,
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
    l2_lambda: float = 1e-3,
    loss_type: str = "mse",
):
    original_state = clone_ttt_adapter_params(model)
    previous_mode_is_training = model.training

    _, ttt_info = ttt_adapt_on_example(
        model=model,
        a=a,
        b=b,
        vocab=vocab,
        representation=representation,
        device=device,
        ttt_steps=ttt_steps,
        ttt_lr=ttt_lr,
        l2_lambda=l2_lambda,
        offset=offset,
        loss_type=loss_type,
    )

    try:
        out_tokens, pred_int = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max_new_tokens,
            offset=offset,
        )
    finally:
        restore_ttt_adapter_params(model, original_state)
        if previous_mode_is_training:
            model.train()
        else:
            model.eval()

    return out_tokens, pred_int, ttt_info


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


def exact_match_accuracy_with_ttt(
    model,
    dataloader,
    vocab: Vocab,
    representation: str,
    device: str = "cpu",
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
    l2_lambda: float = 1e-3,
    loss_type: str = "mse",
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
            _, pred, _ = greedy_generate_with_ttt(
                model=model,
                a=a,
                b=b,
                vocab=vocab,
                representation=representation,
                device=device,
                max_new_tokens=L + 3,
                offset=0,
                ttt_steps=ttt_steps,
                ttt_lr=ttt_lr,
                l2_lambda=l2_lambda,
                loss_type=loss_type,
            )
            if pred == c_true:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


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
    use_ttt: bool = False,
    ttt_steps: int = 5,
    ttt_lr: float = 1e-2,
    l2_lambda: float = 1e-3,
    loss_type: str = "mse",
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

    with torch.no_grad():
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

    if use_ttt:
        em = exact_match_accuracy_with_ttt(
            model=model,
            dataloader=eval_loader,
            vocab=vocab,
            representation=representation,
            device=device,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            l2_lambda=l2_lambda,
            loss_type=loss_type,
        )
    else:
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
    teacher_forcing_decay_until_epoch: int = 100,
):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)

    best_valid_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tf_prob = teacher_forcing_probability(
            epoch=epoch,
            decay_until_epoch=teacher_forcing_decay_until_epoch,
        )

        train_loss_sum = 0.0
        train_acc_sum = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            logits = forward_with_scheduled_sampling(
                model=model,
                input_ids=input_ids,
                target_ids=target_ids,
                position_ids=position_ids,
                pad_id=vocab.PAD_ID,
                teacher_forcing_prob=tf_prob,
            )

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
                        "max_position_id": model.max_position_id,
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

            ood_metrics_no_ttt = evaluate_on_range(
                model=model,
                vocab=vocab,
                representation=representation,
                min_value=1000,
                max_value=9999,
                num_samples=ood_eval_samples,
                batch_size=valid_loader.batch_size,
                device=device,
                seed=1000 + epoch,
                use_ttt=False,
            )

            ood_metrics_ttt = evaluate_on_range(
                model=model,
                vocab=vocab,
                representation=representation,
                min_value=1000,
                max_value=9999,
                num_samples=min(200, ood_eval_samples),
                batch_size=valid_loader.batch_size,
                device=device,
                seed=2000 + epoch,
                use_ttt=True,
                ttt_steps=5,
                ttt_lr=1e-2,
                l2_lambda=1e-3,
                loss_type="mse",
            )

            print(
                f"Epoch {epoch:03d} | "
                f"tf_prob={tf_prob:.3f} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"train_tok_acc={avg_train_acc:.4f} | "
                f"valid_loss={avg_valid_loss:.4f} | "
                f"valid_tok_acc={avg_valid_acc:.4f} | "
                f"valid_EM={valid_em:.4f} | "
                f"OOD_noTTT_loss={ood_metrics_no_ttt['loss']:.4f} | "
                f"OOD_noTTT_tok_acc={ood_metrics_no_ttt['token_accuracy']:.4f} | "
                f"OOD_noTTT_EM={ood_metrics_no_ttt['exact_match']:.4f} | "
                f"OOD_TTT_EM={ood_metrics_ttt['exact_match']:.4f}"
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

    train_min_value = 0
    train_max_value = 999

    ood_min_value = 1000
    ood_max_value = 9999

    train_samples = 20000
    valid_samples = 2000
    ood_eval_samples = 1000

    batch_size =  128
    epochs = 300
    lr = 4e-3

    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 256
    dropout = 0.1

    ttt_hidden_dim = 256
    ttt_dropout = 0.0

    max_pos_offset = 500

    vocab = build_vocab(representation)

    train_loader = build_loader(
        vocab=vocab,
        num_samples=train_samples,
        min_value=train_min_value,
        max_value=train_max_value,
        batch_size=batch_size,
        shuffle=True,
        random_shift_positions=False,
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
        max_pos_offset=100,
        seed=123,
        representation=representation,
    )

    max_ood_operand_len = estimate_max_binary_len(ood_max_value)
    max_position_id = 500

    print("max_ood_operand_len:", max_ood_operand_len)
    print("max_position_id:", max_position_id)

    model = DecoderOnlyAdditionTransformer(
        vocab_size=vocab.VOCAB_SIZE,
        pad_id=vocab.PAD_ID,
        max_position_id=max_position_id + 100,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        ttt_hidden_dim=ttt_hidden_dim,
        ttt_dropout=ttt_dropout,
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
        teacher_forcing_decay_until_epoch=100,
    )

    print("\n=== seen-range checks: 0..999 ===")
    seen_cases = [
        (34, 5),
        (507, 92),
        (999, 999),
        (120, 305),
    ]

    for a, b in seen_cases:
        max_new = max(
            len(int_to_base_string(a, representation)),
            len(int_to_base_string(b, representation)),
        ) + 3

        tokens, pred = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max_new,
            offset=6,
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"a_bin={int_to_base_string(a, representation)} | "
            f"b_bin={int_to_base_string(b, representation)} | "
            f"pred={pred} | tokens={tokens}"
        )

    print("\n=== OOD checks without TTT ===")
    ood_cases = [
        (1000, 1),
        (1234, 5678),
        (9999, 1),
        (4321, 876),
    ]

    for a, b in ood_cases:
        max_new = max(
            len(int_to_base_string(a, representation)),
            len(int_to_base_string(b, representation)),
        ) + 3

        tokens, pred = greedy_generate(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max_new,
            offset=0,
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"a_bin={int_to_base_string(a, representation)} | "
            f"b_bin={int_to_base_string(b, representation)} | "
            f"pred={pred} | tokens={tokens}"
        )

    print("\n=== OOD checks with TTT ===")
    for a, b in ood_cases:
        max_new = max(
            len(int_to_base_string(a, representation)),
            len(int_to_base_string(b, representation)),
        ) + 3

        tokens, pred, ttt_info = greedy_generate_with_ttt(
            model=model,
            a=a,
            b=b,
            vocab=vocab,
            representation=representation,
            device=device,
            max_new_tokens=max_new,
            offset=0,
            ttt_steps=5,
            ttt_lr=1e-2,
            l2_lambda=1e-3,
            loss_type="mse",
        )
        print(
            f"{a} + {b} = {a + b} | "
            f"pred={pred} | "
            f"ttt_loss={ttt_info['final_loss']:.6f} | "
            f"ttt_comm={ttt_info['final_comm_loss']:.6f} | "
            f"tokens={tokens}"
        )

    print("\n=== OOD-range evaluation: 1000..9999 ===")
    ood_metrics_no_ttt = evaluate_on_range(
        model=model,
        vocab=vocab,
        representation=representation,
        min_value=ood_min_value,
        max_value=ood_max_value,
        num_samples=500,
        batch_size=batch_size,
        device=device,
        seed=777,
        use_ttt=False,
    )
    print(
        f"OOD[1000,9999] no TTT | "
        f"loss={ood_metrics_no_ttt['loss']:.4f} | "
        f"tok_acc={ood_metrics_no_ttt['token_accuracy']:.4f} | "
        f"EM={ood_metrics_no_ttt['exact_match']:.4f}"
    )

    ood_metrics_ttt = evaluate_on_range(
        model=model,
        vocab=vocab,
        representation=representation,
        min_value=ood_min_value,
        max_value=ood_max_value,
        num_samples=200,
        batch_size=batch_size,
        device=device,
        seed=888,
        use_ttt=True,
        ttt_steps=5,
        ttt_lr=1e-2,
        l2_lambda=1e-3,
        loss_type="mse",
    )
    print(
        f"OOD[1000,9999] with TTT | "
        f"loss={ood_metrics_ttt['loss']:.4f} | "
        f"tok_acc={ood_metrics_ttt['token_accuracy']:.4f} | "
        f"EM={ood_metrics_ttt['exact_match']:.4f}"
    )


if __name__ == "__main__":
    main()
