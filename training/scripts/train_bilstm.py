"""
Train and evaluate a Bidirectional LSTM (BiLSTM) text classifier on the 17-column review dataset.
Adapted from Pramit Ghosh's packed-sequence BiLSTM architecture (scripts/run_ott_bilstm.py).

Features:
- Bidirectional LSTM with packed padded sequences
- Word vocabulary with frequency thresholding
- Class-weighted BCEWithLogitsLoss for class imbalance handling
- Validation-based checkpointing and early stopping
- Exact evaluation on test split (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
"""

from __future__ import annotations

import json
import logging
import re
import time
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

SEED = 42
MAX_VOCAB = 25000
MIN_FREQ = 2
MAX_LENGTH = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 128
EPOCHS = 3
PATIENCE = 2
LR = 1e-3


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        # hidden shape: (2, batch_size, hidden_dim)
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
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
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
    y_train = get_labels(df_train)

    val_texts = get_text_col(df_val)
    y_val = get_labels(df_val)

    test_texts = get_text_col(df_test)
    y_test = get_labels(df_test)

    logger.info("Building vocabulary from training texts...")
    vocab = build_vocab(train_texts)
    logger.info(f"Vocab size: {len(vocab)}")

    train_ds = ReviewTextDataset(train_texts, y_train, vocab)
    val_ds = ReviewTextDataset(val_texts, y_val, vocab)
    test_ds = ReviewTextDataset(test_texts, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    
    # Calculate pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    patience_counter = 0

    logger.info("Starting BiLSTM training...")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for tokens, lengths, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_train_loss = total_loss / len(train_loader.dataset)
        val_probs, _ = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_val, val_probs)

        logger.info(
            f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), ARTIFACTS_DIR / "best_bilstm.pt")
            logger.info(f"Saved new best model checkpoint with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds.")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_bilstm.pt", map_location=device))

    val_probs, _ = evaluate(model, val_loader, device)
    val_metrics = compute_metrics(y_val, val_probs)

    test_probs, _ = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_test, test_probs)

    logger.info(f"Final BiLSTM Test F1: {test_metrics['f1']:.4f}, Recall: {test_metrics['recall']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")

    np.save(ARTIFACTS_DIR / "test_fake_probs.npy", test_probs)
    np.save(ARTIFACTS_DIR / "val_fake_probs.npy", val_probs)

    # Save vocab
    with open(ARTIFACTS_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f)

    results = {
        "model": "Packed BiLSTM Classifier",
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "vocab_size": len(vocab),
        "train_time_sec": train_time,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACTS_DIR / "bilstm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"BiLSTM artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
