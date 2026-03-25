from typing import List
import torch

class Vocab:
    def __init__(self, mode="decimal"):

        if mode == "decimal":
            digits = [str(i) for i in range(10)]
        else:
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