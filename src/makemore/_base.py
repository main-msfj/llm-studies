"""Make More Implementation"""

from itertools import pairwise

import torch


class Makemore:
    SEP = "."  # token de inicio/fin

    def __init__(self, words: list[str]) -> None:
        self.words = words

        chars = sorted(set("".join(words)))  # caracteres únicos (~26)
        if self.SEP in chars:
            raise ValueError(f"{self.SEP!r} is reserved and cannot appear in words")

        # "." en el índice 0, las letras de 1 en adelante
        self.stoi = {self.SEP: 0} | {s: i + 1 for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def bigram_counts(
        self, start_token: str | None = None, end_token: str | None = None
    ) -> torch.Tensor:
        """N[i, j] = cuántas veces el carácter j sigue al carácter i."""
        st = start_token or self.SEP
        end = end_token or self.SEP

        n = len(self.stoi)
        N = torch.zeros((n, n), dtype=torch.int32)
        for word in self.words:
            chs = [st, *word, end]
            for ch1, ch2 in pairwise(chs):
                N[self.stoi[ch1], self.stoi[ch2]] += 1
        return N
