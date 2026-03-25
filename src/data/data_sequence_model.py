#loading and data data
from typing import List
import torch
from torch.utils.data import Dataset
from src.data.vocab import Vocab


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


