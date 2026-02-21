from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from konlpy.tag import Mecab

from modern.data import build_dataloaders


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hid_dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, hidden = self.rnn(emb)
        logits = self.fc(hidden[-1])
        return logits.squeeze(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = torch.round(torch.sigmoid(logits))
    correct = (preds == labels).float()
    return correct.mean()


def run_epoch(model, loader, criterion, optimizer, device, train=True, max_batches=None):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for step, (x, _lengths, y) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        acc = binary_accuracy(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1

    return total_loss / steps, total_acc / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_data.csv")
    parser.add_argument("--valid-csv", default="data/test_data.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mecab = Mecab()
    tokenizer = mecab.morphs

    train_loader, valid_loader, vocab = build_dataloaders(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        min_freq=2,
    )

    model = SentimentRNN(vocab_size=len(vocab.itos), pad_id=vocab.pad_id).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, max_batches=args.max_batches
        )
        valid_loss, valid_acc = run_epoch(
            model, valid_loader, criterion, optimizer, device, train=False, max_batches=args.max_batches
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )


if __name__ == "__main__":
    main()
