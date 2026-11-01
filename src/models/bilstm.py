"""
Packed Bidirectional LSTM (BiLSTM) text classifier architecture.
Ported from Pramit Ghosh's recurrent neural network pipeline.
"""

import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).casefold())


def encode_sequence(text: str, vocab: dict[str, int], max_length: int = 100) -> list[int]:
    tokens = tokenize_text(text)[:max_length]
    ids = [vocab.get(tok, 1) for tok in tokens]
    return ids if ids else [1]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(self.dropout(representation)).squeeze(1)
        return logits
