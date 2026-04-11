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


@dataclass
class CarryVocab:
    tokens: list
    stoi: dict
    itos: dict
    PAD_ID: int
    ZERO_ID: int
    ONE_ID: int
    VOCAB_SIZE: int


def build_carry_vocab() -> CarryVocab:
    tokens = ["<PAD>", "0", "1"]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return CarryVocab(
        tokens=tokens,
        stoi=stoi,
        itos=itos,
        PAD_ID=stoi["<PAD>"],
        ZERO_ID=stoi["0"],
        ONE_ID=stoi["1"],
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


def build_reversed_sum_and_carry_sequences(a: int, b: int, representation: str):
    """
    Returns:
        tgt_tokens: list[str]        # reversed digits of a+b
        carry_in_tokens: list[str]   # previous carry for each step
        carry_out_tokens: list[str]  # current carry for each step
    """
    if representation == "binary":
        base = 2
    elif representation == "decimal":
        base = 10
    else:
        raise ValueError(f"Unknown representation: {representation}")

    a_rev = int_to_reversed_string(a, representation)
    b_rev = int_to_reversed_string(b, representation)

    max_len = max(len(a_rev), len(b_rev))

    # pad with zeros on reversed representation
    a_digits = [int(ch) for ch in a_rev] + [0] * (max_len - len(a_rev))
    b_digits = [int(ch) for ch in b_rev] + [0] * (max_len - len(b_rev))

    tgt_tokens = []
    carry_in_tokens = []
    carry_out_tokens = []

    carry_prev = 0

    for da, db in zip(a_digits, b_digits):
        carry_in_tokens.append(str(carry_prev))

        s = da + db + carry_prev
        digit = s % base
        carry_now = s // base

        tgt_tokens.append(str(digit))
        carry_out_tokens.append(str(carry_now))

        carry_prev = carry_now

    # if last carry remains, it becomes an extra output digit
    if carry_prev > 0:
        carry_in_tokens.append(str(carry_prev))
        tgt_tokens.append(str(carry_prev))
        carry_out_tokens.append("0")

    return tgt_tokens, carry_in_tokens, carry_out_tokens


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

@dataclass
class SampleWithCarry:
    src_ids: list
    tgt_input_ids: list
    tgt_output_ids: list
    carry_input_ids: list
    carry_output_ids: list
    a: int
    b: int
    sum_: int
    src_text: str
    tgt_text: str
    carry_input_text: str
    carry_output_text: str


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

        if num_digits == 1:
            min_value = 0
            max_value = 9
        elif num_digits == 2:
            min_value = 10
            max_value = 99
        elif num_digits == 3:
            min_value = 0
            max_value = 999
        elif num_digits >= 4:
            min_value = 10 ** (num_digits - 1)
            max_value = 10 ** num_digits - 1
        else:
            raise ValueError("num_digits must be >= 1")

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
                    src_ids=src_ids,
                    tgt_input_ids=tgt_input_ids,
                    tgt_output_ids=tgt_output_ids,
                    a=a,
                    b=b,
                    sum_=s,
                    src_text="".join(src_tokens),
                    tgt_text="".join(tgt_tokens),
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


class AdditionDatasetWithCarry(Dataset):
    """
    Input:
        reversed(padded(a)) + '+' + reversed(padded(b)) + '='

    Token target:
        reversed(padded(a+b))

    Token decoder input:
        <SOS> + target

    Token decoder output:
        target + <EOS>

    Carry decoder input:
        previous carry sequence
        e.g. [0, c0, c1, ...]

    Carry decoder output:
        current carry sequence
        e.g. [c0, c1, c2, ...]
    """

    def __init__(
        self,
        vocab: Vocab,
        carry_vocab: CarryVocab,
        num_samples: int = 20000,
        num_digits: int = 3,
        representation: str = "binary",
    ):
        self.samples = []
        self.num_samples = num_samples
        self.representation = representation
        self.vocab = vocab
        self.carry_vocab = carry_vocab

        if num_digits == 1:
            min_value = 0
            max_value = 9
        elif num_digits == 2:
            min_value = 10
            max_value = 99
        elif num_digits == 3:
            min_value = 0
            max_value = 999
        elif num_digits >= 4:
            min_value = 10 ** (num_digits - 1)
            max_value = 10 ** num_digits - 1
        else:
            raise ValueError("num_digits must be >= 1")

        for _ in range(num_samples):
            a = random.randint(min_value, max_value)
            b = random.randint(min_value, max_value)
            s = a + b

            a_rev, b_rev, tgt_tokens, carry_input_tokens, carry_output_tokens = \
                self._build_padded_example(a, b, representation)

            src_tokens = list(a_rev) + ["+"] + list(b_rev) + ["="]
            src_ids = encode_tokens(src_tokens, vocab)

            tgt_input_ids = [vocab.SOS_ID] + encode_tokens(tgt_tokens, vocab)
            tgt_output_ids = encode_tokens(tgt_tokens, vocab) + [vocab.EOS_ID]

            carry_input_ids = [carry_vocab.stoi[t] for t in carry_input_tokens] + [carry_vocab.PAD_ID]
            carry_output_ids = [carry_vocab.stoi[t] for t in carry_output_tokens] + [carry_vocab.PAD_ID]

            self.samples.append(
                SampleWithCarry(
                    src_ids=src_ids,
                    tgt_input_ids=tgt_input_ids,
                    tgt_output_ids=tgt_output_ids,
                    carry_input_ids=carry_input_ids,
                    carry_output_ids=carry_output_ids,
                    a=a,
                    b=b,
                    sum_=s,
                    src_text="".join(src_tokens),
                    tgt_text="".join(tgt_tokens),
                    carry_input_text="".join(carry_input_tokens),
                    carry_output_text="".join(carry_output_tokens),
                )
            )

    @staticmethod
    def _to_base_string(n: int, representation: str) -> str:
        if representation == "binary":
            return bin(n)[2:]
        elif representation == "decimal":
            return str(n)
        else:
            raise ValueError(f"Unknown representation: {representation}")

    @classmethod
    def _pad_number(cls, n: int, total_len: int, representation: str) -> str:
        return cls._to_base_string(n, representation).zfill(total_len)

    @classmethod
    def _build_padded_example(cls, a: int, b: int, representation: str):
        if representation == "binary":
            base = 2
        elif representation == "decimal":
            base = 10
        else:
            raise ValueError(f"Unknown representation: {representation}")

        a_str = cls._to_base_string(a, representation)
        b_str = cls._to_base_string(b, representation)

        L = max(len(a_str), len(b_str))

        a_pad = a_str.zfill(L)
        b_pad = b_str.zfill(L)

        s_pad = cls._to_base_string(a + b, representation).zfill(L + 1)

        a_rev = a_pad[::-1]
        b_rev = b_pad[::-1]
        s_rev = s_pad[::-1]

        a_digits = [int(ch) for ch in a_rev]
        b_digits = [int(ch) for ch in b_rev]
        tgt_tokens = list(s_rev)

        carry_input_tokens = []
        carry_output_tokens = []

        carry_prev = 0
        for da, db in zip(a_digits, b_digits):
            carry_input_tokens.append(str(carry_prev))
            total = da + db + carry_prev
            carry_now = total // base
            carry_output_tokens.append(str(carry_now))
            carry_prev = carry_now

        carry_input_tokens.append(str(carry_prev))
        carry_output_tokens.append("0")

        return a_rev, b_rev, tgt_tokens, carry_input_tokens, carry_output_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "src_ids": torch.tensor(s.src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(s.tgt_input_ids, dtype=torch.long),
            "tgt_output_ids": torch.tensor(s.tgt_output_ids, dtype=torch.long),
            "carry_input_ids": torch.tensor(s.carry_input_ids, dtype=torch.long),
            "carry_output_ids": torch.tensor(s.carry_output_ids, dtype=torch.long),
            "a": s.a,
            "b": s.b,
            "sum_": s.sum_,
            "src_text": s.src_text,
            "tgt_text": s.tgt_text,
            "carry_input_text": s.carry_input_text,
            "carry_output_text": s.carry_output_text,
        }