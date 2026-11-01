"""
Evaluate saved BiLSTM checkpoint on validation and test splits and save metrics.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VAL_PATH = ROOT / "data" / "splits" / "validation.csv"
TEST_PATH = ROOT / "data" / "splits" / "test.csv"
ARTIFACTS_DIR = ROOT / "training" / "artifacts" / "bilstm"

MAX_VOCAB = 25000
MIN_FREQ = 2
MAX_LENGTH = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 128


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).casefold())


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    tokens = [
        token
        for token, count in counts.most_common()
        if count >= MIN_FREQ and token not in vocab
    ][: MAX_VOCAB - len(vocab)]
    vocab.update({token: idx + 2 for idx, token in enumerate(tokens)})
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    tokens = tokenize(text)[:MAX_LENGTH]
    ids = [vocab.get(tok, 1) for tok in tokens]
    return ids if ids else [1]


class ReviewTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: dict[str, int]):
        self.sequences = [encode_text(t, vocab) for t in texts]
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        length = min(len(seq), MAX_LENGTH)
        padded = seq[:MAX_LENGTH] + [0] * max(0, MAX_LENGTH - len(seq))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(max(1, length), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.4):
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            logits = model(tokens, lengths)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    return probs, all_labels


def main():
    device = torch.device("cpu")
    logger.info("Loading splits...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    def get_text_col(df: pd.DataFrame) -> list[str]:
        col = "review_text" if "review_text" in df.columns else "text"
        return df[col].fillna("").astype(str).tolist()

    def get_labels(df: pd.DataFrame) -> np.ndarray:
        series = df["label"].astype(str).str.lower().str.strip()
        return (series.isin(["fake", "1", "cg"])).values.astype(int)

    train_texts = get_text_col(df_train)
    val_texts = get_text_col(df_val)
    test_texts = get_text_col(df_test)

    y_val = get_labels(df_val)
    y_test = get_labels(df_test)

    vocab = build_vocab(train_texts)
    logger.info(f"Vocab size: {len(vocab)}")

    val_ds = ReviewTextDataset(val_texts, y_val, vocab)
    test_ds = ReviewTextDataset(test_texts, y_test, vocab)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_bilstm.pt", map_location=device))
    logger.info("Loaded best_bilstm.pt successfully.")

    val_probs, _ = evaluate(model, val_loader, device)
    val_metrics = compute_metrics(y_val, val_probs, threshold=0.50)
    logger.info(f"Val F1: {val_metrics['f1']:.4f}, Recall: {val_metrics['recall']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

    test_probs, _ = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_test, test_probs, threshold=0.50)
    logger.info(f"Test F1: {test_metrics['f1']:.4f}, Recall: {test_metrics['recall']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")

    np.save(ARTIFACTS_DIR / "test_fake_probs.npy", test_probs)
    np.save(ARTIFACTS_DIR / "val_fake_probs.npy", val_probs)

    results = {
        "model": "Packed BiLSTM Classifier",
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "vocab_size": len(vocab),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACTS_DIR / "bilstm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(ARTIFACTS_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    logger.info("BiLSTM metrics and vocab.json saved.")


if __name__ == "__main__":
    main()
