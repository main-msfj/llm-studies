"""Tests for the dataset and dataloader in src.tokenizer."""

import torch

from src.tokenizer import GPTDataset, ToyDataset, create_dataloader


class CharTokenizer:
    """Fake tokenizer: one token per character, so no downloads are needed."""

    def __init__(self):
        self.allowed_special_seen = None

    def encode(self, text, allowed_special):
        self.allowed_special_seen = allowed_special
        return [ord(c) for c in text]


TEXT = "abcdefghijklmnopqrst"  # 20 tokens


def test_window_count_without_overlap():
    ds = GPTDataset(TEXT, CharTokenizer(), max_length=4, stride=4, allowed_special=set())
    assert len(ds) == 4  # range(0, 20 - 4, 4) -> 0, 4, 8, 12


def test_smaller_stride_gives_more_windows():
    ds = GPTDataset(TEXT, CharTokenizer(), max_length=4, stride=2, allowed_special=set())
    assert len(ds) == 8


def test_target_is_input_shifted_by_one():
    ds = GPTDataset(TEXT, CharTokenizer(), max_length=4, stride=4, allowed_special=set())
    x, y = ds[0]
    assert x.tolist() == [ord(c) for c in "abcd"]
    assert y.tolist() == [ord(c) for c in "bcde"]


def test_dataloader_batch_shapes():
    dl = create_dataloader(
        TEXT, batch_size=2, max_length=4, stride=4, shuffle=False, tokenizer=CharTokenizer()
    )
    x, y = next(iter(dl))
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert len(dl) == 2  # 4 windows, batch of 2


def test_create_dataloader_passes_allowed_special():
    tok = CharTokenizer()
    create_dataloader(TEXT, batch_size=2, max_length=4, stride=4, tokenizer=tok)
    assert tok.allowed_special_seen == "all"


def test_toy_dataset():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([0, 1])
    ds = ToyDataset(X, y)
    assert len(ds) == 2
    x0, y0 = ds[1]
    assert x0.tolist() == [3.0, 4.0]
    assert y0.item() == 1
