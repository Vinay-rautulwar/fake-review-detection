"""Reproduce text-only baselines on the Ott deceptive-opinion corpus.

This experiment deliberately uses text only. The corpus has no reviewer history
or review graph, so it cannot evaluate FakeGuard's behavioral or graph views.
Run from the project root with: python scripts/run_ott_experiments.py
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "deceptive-opinion.csv"
OUTPUT_DIR = ROOT / "results" / "ott_experiments"
SEED = 42
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "fake_precision",
    "fake_recall",
    "fake_f1",
    "roc_auc",
    "average_precision",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    conflicting = frame.groupby("normalized_text")["label"].nunique().gt(1)
    if conflicting.any():
        raise ValueError("Exact normalized duplicate texts have conflicting labels")

    duplicate_count = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    if frame["hotel"].nunique() < 5:
        raise ValueError("At least five hotel groups are required for grouped CV")
    return frame, duplicate_count


def make_models() -> dict[str, Pipeline]:
    vectorizer_args = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    return {
        "Logistic Regression (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_args)),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Linear SVM (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_args)),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                ),
            ]
        ),
        "Multinomial NB (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_args)),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
        "MLP (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_args)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=100,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def get_scores(model: Pipeline, test_texts: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_texts)
        positive_index = list(model.classes_).index(1)
        return probabilities[:, positive_index]
    return np.asarray(model.decision_function(test_texts))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fake_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Ott corpus not found: {DATA_PATH}")

    frame, duplicate_count = load_corpus()
    texts = frame["text"]
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
        for model_name, template in make_models().items():
            for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
                print(
                    f"Fitting {protocol_name} | {model_name} | fold {fold_number}/5",
                    flush=True,
                )
                model = sklearn.base.clone(template)
                train_texts = texts.iloc[train_indices]
                test_texts = texts.iloc[test_indices]
                y_train = labels[train_indices]
                y_test = labels[test_indices]

                model.fit(train_texts, y_train)
                predictions = model.predict(test_texts)
                scores = get_scores(model, test_texts)
                metric_values = compute_metrics(y_test, predictions, scores)

                test_hotels = sorted(set(groups[test_indices]))
                records.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "fold": fold_number,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "n_fake_test": int(y_test.sum()),
                        "test_hotels": ",".join(test_hotels),
                        **metric_values,
                    }
                )
                for index in train_indices:
                    split_records.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold_number,
                            "row_index": int(index),
                            "role": "train",
                            "hotel": groups[index],
                            "label": int(labels[index]),
                        }
                    )
                for index in test_indices:
                    split_records.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold_number,
                            "row_index": int(index),
                            "role": "test",
                            "hotel": groups[index],
                            "label": int(labels[index]),
                        }
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(records)
    fold_frame.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(split_records).drop_duplicates().to_csv(
        OUTPUT_DIR / "split_manifest.csv", index=False
    )

    summary_rows = []
    for (protocol_name, model_name), group_frame in fold_frame.groupby(
        ["protocol", "model"], sort=False
    ):
        row = {
            "protocol": protocol_name,
            "model": model_name,
            "folds": len(group_frame),
        }
        for metric_name in METRIC_NAMES:
            row[f"{metric_name}_mean"] = group_frame[metric_name].mean()
            row[f"{metric_name}_std"] = group_frame[metric_name].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, float_format="%.6f")

    metadata = {
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATA_PATH),
        "source_references": [
            "Ott et al., ACL 2011, https://aclanthology.org/P11-1032/",
            "Ott et al., NAACL 2013, https://aclanthology.org/N13-1053/",
        ],
        "task": "deceptive (1) versus truthful (0) hotel review text classification",
        "raw_rows": int(pd.read_csv(DATA_PATH).shape[0]),
        "rows_after_exact_normalized_text_deduplication": int(len(frame)),
        "duplicates_removed": duplicate_count,
        "labels_after_deduplication": {
            "truthful_0": int((labels == 0).sum()),
            "deceptive_1": int((labels == 1).sum()),
        },
        "groups": {"name": "hotel", "count": int(frame["hotel"].nunique())},
        "random_seed": SEED,
        "protocols": {
            "stratified_random_5fold": "StratifiedKFold(shuffle=True, random_state=42); hotel overlap is allowed",
            "heldout_hotel_5fold": "GroupKFold(5); no hotel appears in both train and test within a fold",
        },
        "text_pipeline": {
            "features": "word TF-IDF",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 10000,
            "sublinear_tf": True,
            "stop_words": None,
            "no_behavior_or_graph_features": True,
        },
        "models": [
            "LogisticRegression(C=1,class_weight=balanced,max_iter=2000)",
            "LinearSVC(C=1,class_weight=balanced)",
            "MultinomialNB(alpha=0.5)",
            "MLPClassifier(hidden_layer_sizes=(32,),alpha=1e-4,batch_size=64,learning_rate_init=1e-3,max_iter=100,early_stopping=True,n_iter_no_change=15)",
        ],
        "metrics": list(METRIC_NAMES),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_limit": "This controlled corpus supports text-only deception classification; it cannot validate reviewer behavior, graph features, or platform deployment.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Corpus rows after deduplication: {len(frame)}; removed duplicates: {duplicate_count}")
    print(f"Hotels: {frame['hotel'].nunique()}; label counts: {frame['deceptive'].value_counts().to_dict()}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved fold metrics, split manifest, and metadata under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
