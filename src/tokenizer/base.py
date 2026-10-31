"""Tokenizer for LLM models."""

import typing as t

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """Dataset for GPT Models."""

    def __init__(
        self,
        data: object,
        tokenizer: t.Any,
        max_length: int,
        stride: int,
        allowed_special: set | str,
    ):
        self.input_ids = []
        self.target_ids = []

        # Tokenize Data
        token_ids = tokenizer.encode(data, allowed_special=allowed_special)

        # Sliding Window Approach
        # Total tokens - max length for a intreval of stride
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    data: object,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: t.Any | None = None,
    allowed_special: set | str = "all",
) -> DataLoader:
    """Create a DataLoader for GPT Models."""
    # Initialize Tokenizer
    tokenizer = tokenizer or tiktoken.get_encoding("gpt2")

    # Create Dataset
    dataset = GPTDataset(data, tokenizer, max_length, stride, allowed_special)

    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
