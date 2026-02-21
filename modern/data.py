from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]


class NSMCDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: Callable[[str], list[str]], vocab: Vocab):
        df = pd.read_csv(csv_path)
        self.texts = df["document"].fillna("").astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab.stoi.get(token, self.vocab.unk_id) for token in tokens]
        if not ids:
            ids = [self.vocab.unk_id]
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


def build_vocab(
    texts: Iterable[str], tokenizer: Callable[[str], list[str]], min_freq: int = 1
) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenizer(text))

    itos = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in counter.items():
        if freq >= min_freq:
            itos.append(token)

    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def make_collate_fn(pad_id: int):
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        xs, ys = zip(*batch)
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        padded_x = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        labels = torch.stack(ys)
        return padded_x, lengths, labels

    return collate_fn


def build_dataloaders(
    train_csv: str,
    valid_csv: str,
    tokenizer: Callable[[str], list[str]],
    batch_size: int = 64,
    min_freq: int = 1,
):
    train_df = pd.read_csv(train_csv)
    vocab = build_vocab(train_df["document"].fillna("").astype(str).tolist(), tokenizer, min_freq)

    train_dataset = NSMCDataset(train_csv, tokenizer, vocab)
    valid_dataset = NSMCDataset(valid_csv, tokenizer, vocab)

    collate_fn = make_collate_fn(vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, vocab
