"""Train a reproducible BiLSTM text baseline on the Ott review corpus.

This is a text-only experiment. It cannot evaluate behavioral or graph views.
Run from the project root with: python scripts/run_ott_bilstm.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "deceptive-opinion.csv"
OUTPUT_DIR = ROOT / "results" / "ott_experiments"
SEED = 42
MAX_VOCAB = 20000
MIN_TOKEN_FREQUENCY = 2
MAX_LENGTH = 200
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def load_corpus() -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(DATA_PATH)
    required = {"text", "deceptive", "hotel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[frame["deceptive"].isin(["deceptive", "truthful"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = (frame["deceptive"] == "deceptive").astype(int)
    frame["normalized_text"] = (
        frame["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    if frame.groupby("normalized_text")["label"].nunique().gt(1).any():
        raise ValueError("Exact normalized duplicate texts have conflicting labels")
    duplicate_count = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    return frame, duplicate_count


def make_vocab(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    tokens = [
        token
        for token, count in counts.most_common()
        if count >= MIN_TOKEN_FREQUENCY and token not in vocab
    ][: MAX_VOCAB - len(vocab)]
    vocab.update({token: index + 2 for index, token in enumerate(tokens)})
    return vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:MAX_LENGTH]]
    return ids or [1]


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: dict[str, int]):
        self.sequences = [encode(text, vocab) for text in texts]
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        padded = sequence + [0] * (MAX_LENGTH - len(sequence))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(len(sequence), dtype=torch.long),
            torch.tensor(self.labels[index]),
        )


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(2 * HIDDEN_DIM, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation)).squeeze(1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for tokens, lengths, labels in loader:
        tokens = tokens.to(device)
        lengths = lengths.cpu()
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(tokens, lengths)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(logits_all),
        np.concatenate(labels_all).astype(int),
    )


def fit_and_score(
    train_texts: list[str],
    train_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], int, int]:
    seed_everything(seed)
    fit_indices, validation_indices = train_test_split(
        np.arange(len(train_texts)),
        test_size=0.15,
        random_state=seed,
        stratify=train_labels,
    )
    fitting_texts = [train_texts[index] for index in fit_indices]
    fitting_labels = train_labels[fit_indices]
    validation_texts = [train_texts[index] for index in validation_indices]
    validation_labels = train_labels[validation_indices]

    # Vocabulary is fit only on the inner training subset, never on validation or test.
    vocab = make_vocab(fitting_texts)
    train_data = ReviewDataset(fitting_texts, fitting_labels, vocab)
    validation_data = ReviewDataset(validation_texts, validation_labels, vocab)
    test_data = ReviewDataset(test_texts, test_labels, vocab)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = BiLSTMClassifier(len(vocab)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_loss, _, _ = run_epoch(model, validation_loader, loss_fn, device)
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, test_logits, labels = run_epoch(model, test_loader, loss_fn, device)
    scores = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -40, 40)))
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "fake_precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "fake_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "fake_f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
    }
    return metrics, best_epoch, len(vocab)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Ott corpus not found: {DATA_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame, duplicate_count = load_corpus()
    texts = frame["text"].tolist()
    labels = frame["label"].to_numpy()
    groups = frame["hotel"].to_numpy()
    protocols = {
        "stratified_random_5fold": list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(texts, labels)
        ),
        "heldout_hotel_5fold": list(GroupKFold(n_splits=5).split(texts, labels, groups)),
    }
    records: list[dict] = []
    split_records: list[dict] = []
    for protocol_name, splits in protocols.items():
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            fold_seed = SEED + fold
            print(f"Fitting {protocol_name} | BiLSTM | fold {fold}/5", flush=True)
            metrics, best_epoch, vocab_size = fit_and_score(
                [texts[index] for index in train_indices],
                labels[train_indices],
                [texts[index] for index in test_indices],
                labels[test_indices],
                fold_seed,
                device,
            )
            records.append(
                {
                    "protocol": protocol_name,
                    "model": "BiLSTM (random 100d embeddings)",
                    "fold": fold,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "n_fake_test": int(labels[test_indices].sum()),
                    "test_hotels": ",".join(sorted(set(groups[test_indices]))),
                    "best_epoch": best_epoch,
                    "vocabulary_size": vocab_size,
                    **metrics,
                }
            )
            for index in train_indices:
                split_records.append(
                    {"protocol": protocol_name, "fold": fold, "row_index": int(index), "role": "train", "hotel": groups[index], "label": int(labels[index])}
                )
            for index in test_indices:
                split_records.append(
                    {"protocol": protocol_name, "fold": fold, "row_index": int(index), "role": "test", "hotel": groups[index], "label": int(labels[index])}
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(records)
    fold_frame.to_csv(OUTPUT_DIR / "bilstm_fold_metrics.csv", index=False)
    pd.DataFrame(split_records).drop_duplicates().to_csv(
        OUTPUT_DIR / "bilstm_split_manifest.csv", index=False
    )
    metric_names = (
        "accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "roc_auc", "average_precision"
    )
    summary_rows = []
    for (protocol, model_name), group in fold_frame.groupby(["protocol", "model"], sort=False):
        row = {"protocol": protocol, "model": model_name, "folds": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        row["best_epoch_mean"] = group["best_epoch"].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "bilstm_summary.csv", index=False, float_format="%.6f")
    metadata = {
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATA_PATH),
        "source_references": [
            "Ott et al., ACL 2011, https://aclanthology.org/P11-1032/",
            "Ott et al., NAACL 2013, https://aclanthology.org/N13-1053/",
        ],
        "task": "deceptive (1) versus truthful (0) hotel review text classification",
        "raw_rows": int(pd.read_csv(DATA_PATH).shape[0]),
        "rows_after_exact_normalized_text_deduplication": len(frame),
        "duplicates_removed": duplicate_count,
        "groups": {"name": "hotel", "count": int(frame["hotel"].nunique())},
        "random_seed": SEED,
        "protocols": {
            "stratified_random_5fold": "StratifiedKFold(shuffle=True, random_state=42); hotel overlap is allowed",
            "heldout_hotel_5fold": "GroupKFold(5); no hotel appears in both train and test within a fold",
            "inner_validation": "15% stratified split of each training fold; used for early stopping only",
        },
        "model": "single-layer bidirectional LSTM with packed sequences, 100-dimensional randomly initialized embedding, 64 hidden units per direction, dropout 0.5, binary cross-entropy, Adam learning rate 0.001",
        "tokenization": {
            "pattern": "[a-z0-9]+(?:'[a-z0-9]+)?",
            "max_vocabulary": MAX_VOCAB,
            "minimum_token_frequency": MIN_TOKEN_FREQUENCY,
            "maximum_sequence_length": MAX_LENGTH,
            "vocabulary_fit": "training subset only, independently within each fold",
        },
        "training": {"batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "threshold": 0.5},
        "metrics": list(metric_names),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_limit": "Text-only small controlled corpus; does not validate behavioral/graph views or deployment; one seed and no pretrained language model.",
    }
    (OUTPUT_DIR / "bilstm_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
