#loading and preprocessing data
from typing import List
import torch
from torch.utils.data import Dataset

class Vocab:
    def __init__(self, mode="decimal"):
        if mode == "decimal":
            digits = [str(i) for i in range(10)]
        elif mode == "binary":
            digits = ["0","1"]

        self.symbol2id = {}
        self.id2symbol = {}

        for i, d in enumerate(digits):
            self.symbol2id[d] = i
            self.id2symbol[i] = d

        self.EQUAL_ID = len(self.symbol2id)
        self.symbol2id["="] = self.EQUAL_ID
        self.id2symbol[self.EQUAL_ID] = "="

        self.PLUS_ID = len(self.symbol2id)
        self.symbol2id["+"] = self.PLUS_ID
        self.id2symbol[self.PLUS_ID] = "+"

        self.EOS_ID = len(self.symbol2id)
        self.symbol2id["<EOS>"] = self.EOS_ID
        self.id2symbol[self.EOS_ID] = "<EOS>"

        self.PAD_ID = len(self.symbol2id)
        self.symbol2id["<PAD>"] = self.PAD_ID
        self.id2symbol[self.PAD_ID] = "<PAD>"

    def encode(self, example: str) -> List[int]:
        result = []

        for symbol in example:
            result.append(self.symbol2id[symbol])

        if example[-1] != "=":
            result.append(self. EOS_ID)

        return result

    def decode(self, example: torch.Tensor) -> str:
        result = ""
        for symbol in example:
            result += self.id2symbol[symbol.item()]

        return result


def collate_fn(examples: List[str], vocab: Vocab):
    encoded_examples = []
    targets = []
    max_length = 0

    for example in examples:
        if len(example) > max_length:
            max_length = len(example)

    max_length += 1

    for example in examples:
        enc_ex = vocab.encode(example)
        enc_ex.extend([vocab.PAD_ID] * (max_length - len(enc_ex)))
        encoded_examples.append(
            torch.tensor(enc_ex, dtype=torch.int))

        equal_id = enc_ex.index(vocab.EQUAL_ID)
        target = enc_ex.copy()
        target[:equal_id+1] = [vocab.PAD_ID] * (equal_id + 1)
        targets.append(
            torch.tensor(target, dtype= torch.long))

    return (torch.stack(encoded_examples, dim=0),
            torch.stack(targets, dim=0))


def collate_fn_ab_ba(examples: List[str], vocab: Vocab):
    encoded_examples_ab = []
    encoded_examples_ba = []
    targets = []
    max_length = 0

    for example in examples:
        if len(example) > max_length:
            max_length = len(example)

    max_length += 1

    for example in examples:
        enc_ex = vocab.encode(example)
        enc_ex.extend([vocab.PAD_ID] * (max_length - len(enc_ex)))
        encoded_examples_ab.append(
            torch.tensor(enc_ex, dtype=torch.int))

        equal_id = enc_ex.index(vocab.EQUAL_ID)
        plus_id = enc_ex.index(vocab.PLUS_ID)
        enc_ex_ab = enc_ex.copy()
        enc_ex_ab[:plus_id], enc_ex_ab[plus_id+1:equal_id] = enc_ex_ab[plus_id+1:equal_id], enc_ex_ab[:plus_id]
        encoded_examples_ba.append(
            torch.tensor(enc_ex_ab, dtype=torch.int))

        target = enc_ex.copy()
        target[:equal_id+1] = [vocab.PAD_ID] * (equal_id + 1)
        targets.append(
            torch.tensor(target, dtype= torch.long))


    return (torch.stack(encoded_examples_ab, dim=0),
            torch.stack(encoded_examples_ba, dim=0),
            torch.stack(targets, dim=0))

class DatasetSum(Dataset):
    def __init__(self, data: List[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


