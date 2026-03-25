from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import random

@dataclass
class Vocab:
    tokens: list
    stoi: dict
    itos: dict
    PAD_ID: int
    SOS_ID: int
    EOS_ID: int
    VOCAB_SIZE: int


def build_vocab(representation: str) -> Vocab:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    if representation == "binary":
        digits = ["0", "1"]
    elif representation == "decimal":
        digits = [str(i) for i in range(10)]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + digits + ["+", "="]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for tok, i in stoi.items()}

    return Vocab(
        tokens=tokens,
        stoi=stoi,
        itos=itos,
        PAD_ID=stoi[PAD_TOKEN],
        SOS_ID=stoi[SOS_TOKEN],
        EOS_ID=stoi[EOS_TOKEN],
        VOCAB_SIZE=len(tokens),
    )


def int_to_reversed_binary(n: int) -> str:
    return bin(n)[2:][::-1]


def int_to_reversed_decimal(n: int) -> str:
    return str(n)[::-1]


def int_to_reversed_string(n: int, representation: str) -> str:
    if representation == "binary":
        return int_to_reversed_binary(n)
    elif representation == "decimal":
        return int_to_reversed_decimal(n)
    else:
        raise ValueError(f"Unknown representation: {representation}")


def encode_tokens(tokens, vocab: Vocab):
    return [vocab.stoi[t] for t in tokens]


def decode_tokens(ids, vocab: Vocab):
    out = []
    for idx in ids:
        tok = vocab.itos[int(idx)]
        if tok in ["<PAD>", "<SOS>", "<EOS>"]:
            continue
        out.append(tok)
    return out


def parse_tokens_to_int(tokens, representation: str):
    if len(tokens) == 0:
        return 0

    normal = "".join(tokens[::-1])

    if representation == "binary":
        return int(normal, 2)
    elif representation == "decimal":
        return int(normal, 10)
    else:
        raise ValueError(f"Unknown representation: {representation}")


def build_src_ids_from_numbers(a: int, b: int, vocab: Vocab, representation: str):
    a_rev = int_to_reversed_string(a, representation)
    b_rev = int_to_reversed_string(b, representation)
    src_tokens = list(a_rev) + ["+"] + list(b_rev) + ["="]
    src_ids = encode_tokens(src_tokens, vocab)
    src_len = len(src_ids)
    return (
        torch.tensor(src_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor([src_len], dtype=torch.long),
        "".join(src_tokens),
    )


@dataclass
class Sample:
    src_ids: list
    tgt_input_ids: list
    tgt_output_ids: list
    a: int
    b: int
    sum_: int
    src_text: str
    tgt_text: str

class AdditionDataset(Dataset):
    """
    Input:
        reversed(a) + '+' + reversed(b) + '='
    Target:
        reversed(a+b)

    Decoder input:
        <SOS> + target

    Decoder output:
        target + <EOS>
    """
    def __init__(
        self, vocab,
            num_samples: int = 20000,
            num_digits: int = 3,
            representation: str = "binary"):
        self.samples = []
        self.num_samples = num_samples
        self.representation = representation
        self.vocab = vocab

        if num_digits == 3:
            min_value = 0
            max_value = 999
        elif num_digits >= 4:
            min_value = 10 ** (num_digits - 1)
            max_value = 10 ** num_digits - 1
        else:
            raise ValueError("num_digits must be >= 3")

        for _ in range(num_samples):
            a = random.randint(min_value, max_value)
            b = random.randint(min_value, max_value)
            s = a + b

            a_rev = int_to_reversed_string(a, representation)
            b_rev = int_to_reversed_string(b, representation)
            s_rev = int_to_reversed_string(s, representation)

            src_tokens = list(a_rev) + ["+"] + list(b_rev) + ["="]
            tgt_tokens = list(s_rev)

            src_ids = encode_tokens(src_tokens, vocab)
            tgt_input_ids = [vocab.SOS_ID] + encode_tokens(tgt_tokens, vocab)
            tgt_output_ids = encode_tokens(tgt_tokens, vocab) + [vocab.EOS_ID]

            self.samples.append(
                Sample(
                    src_ids=src_ids, #id with plus start and etc
                    tgt_input_ids=tgt_input_ids, #encoded reversed with SOS
                    tgt_output_ids=tgt_output_ids, #encoded reversed with EOS
                    a=a, #original
                    b=b, #original
                    sum_=s,#original
                    src_text="".join(src_tokens), #withiut encoding
                    tgt_text="".join(tgt_tokens), #without encoding
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "src_ids": torch.tensor(s.src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(s.tgt_input_ids, dtype=torch.long),
            "tgt_output_ids": torch.tensor(s.tgt_output_ids, dtype=torch.long),
            "a": s.a,
            "b": s.b,
            "sum_": s.sum_,
            "src_text": s.src_text,
            "tgt_text": s.tgt_text,
        }