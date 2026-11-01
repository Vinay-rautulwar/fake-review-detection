"""Test cross-polarity transfer with hotel-disjoint folds on the Ott corpus.

This is an exploratory text-only stress test: train on one sentiment polarity
and evaluate on either the same or the opposite polarity, while also holding
out hotels. It does not estimate platform-level fraud performance.

Run from the project root with: python scripts/run_ott_polarity_transfer.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
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
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "deceptive-opinion.csv"
OUTPUT_DIR = ROOT / "results" / "ott_experiments"
SEED = 42
POLARITIES = ("positive", "negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus() -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(DATA_PATH)
    required = {"text", "deceptive", "hotel", "polarity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[
        frame["deceptive"].isin(["deceptive", "truthful"])
        & frame["polarity"].isin(POLARITIES)
    ].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = (frame["deceptive"] == "deceptive").astype(int)
    frame["normalized_text"] = (
        frame["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    conflicts = frame.groupby("normalized_text")["label"].nunique().gt(1)
    if conflicts.any():
        raise ValueError("Exact normalized duplicate texts have conflicting labels")
    duplicates = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    return frame, duplicates


def make_models() -> dict[str, Pipeline]:
    vectorizer = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer)),
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
        "Linear SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer)),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                ),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", TfidfVectorizer(**vectorizer)), ("classifier", MultinomialNB(alpha=0.5))]
        ),
    }


def scores_for(model: Pipeline, texts: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        positive_index = list(model.classes_).index(1)
        return probabilities[:, positive_index]
    return np.asarray(model.decision_function(texts))


def main() -> None:
    frame, duplicates = load_corpus()
    if frame["hotel"].nunique() != 20:
        raise ValueError("Expected the 20 hotel groups in the local Ott-style corpus")
    folds = list(GroupKFold(n_splits=5).split(frame, frame["label"], frame["hotel"]))
    models = make_models()
    rows: list[dict] = []
    manifest: list[dict] = []

    # Each held-out set is the same four hotels across within- and cross-polarity
    # conditions, so the conditions are paired by hotel fold.
    for source_polarity in POLARITIES:
        for fold_number, (train_group_indices, test_group_indices) in enumerate(folds, start=1):
            train_hotels = set(frame.iloc[train_group_indices]["hotel"])
            test_hotels = set(frame.iloc[test_group_indices]["hotel"])
            for target_polarity in POLARITIES:
                condition = "within_polarity" if target_polarity == source_polarity else "cross_polarity"
                train_mask = frame["hotel"].isin(train_hotels) & (frame["polarity"] == source_polarity)
                test_mask = frame["hotel"].isin(test_hotels) & (frame["polarity"] == target_polarity)
                train_frame = frame.loc[train_mask]
                test_frame = frame.loc[test_mask]
                y_train = train_frame["label"].to_numpy()
                y_test = test_frame["label"].to_numpy()
                if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
                    raise ValueError("A train/test condition is missing one label class")

                for model_name, template in models.items():
                    model = sklearn.base.clone(template)
                    model.fit(train_frame["text"], y_train)
                    predictions = model.predict(test_frame["text"])
                    scores = scores_for(model, test_frame["text"])
                    rows.append(
                        {
                            "source_polarity": source_polarity,
                            "target_polarity": target_polarity,
                            "condition": condition,
                            "fold": fold_number,
                            "model": model_name,
                            "n_train": len(train_frame),
                            "n_test": len(test_frame),
                            "n_deceptive_test": int(y_test.sum()),
                            "test_hotels": ",".join(sorted(test_hotels)),
                            "accuracy": accuracy_score(y_test, predictions),
                            "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
                            "deceptive_precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
                            "deceptive_recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
                            "deceptive_f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
                            "roc_auc": roc_auc_score(y_test, scores),
                            "average_precision": average_precision_score(y_test, scores),
                        }
                    )
                    for idx in train_frame.index:
                        manifest.append(
                            {"source_polarity": source_polarity, "target_polarity": target_polarity, "fold": fold_number, "role": "train", "row_index": int(idx), "hotel": frame.loc[idx, "hotel"], "polarity": frame.loc[idx, "polarity"], "label": int(frame.loc[idx, "label"])}
                        )
                    for idx in test_frame.index:
                        manifest.append(
                            {"source_polarity": source_polarity, "target_polarity": target_polarity, "fold": fold_number, "role": "test", "row_index": int(idx), "hotel": frame.loc[idx, "hotel"], "polarity": frame.loc[idx, "polarity"], "label": int(frame.loc[idx, "label"])}
                        )
            print(f"Completed source polarity {source_polarity}, hotel fold {fold_number}/5", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(OUTPUT_DIR / "polarity_transfer_fold_metrics.csv", index=False)
    pd.DataFrame(manifest).drop_duplicates().to_csv(
        OUTPUT_DIR / "polarity_transfer_split_manifest.csv", index=False
    )
    metric_names = (
        "accuracy",
        "macro_f1",
        "deceptive_precision",
        "deceptive_recall",
        "deceptive_f1",
        "roc_auc",
        "average_precision",
    )
    summary_rows: list[dict] = []
    group_columns = ["source_polarity", "condition", "target_polarity", "model"]
    for group_keys, group in fold_frame.groupby(group_columns, sort=False):
        result = dict(zip(group_columns, group_keys))
        result["folds"] = len(group)
        for metric in metric_names:
            result[f"{metric}_mean"] = group[metric].mean()
            result[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "polarity_transfer_summary.csv", index=False, float_format="%.6f")
    metadata = {
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATA_PATH),
        "source_references": [
            "Ott et al., ACL 2011, https://aclanthology.org/P11-1032/",
            "Ott et al., NAACL 2013, https://aclanthology.org/N13-1053/",
        ],
        "rows_after_exact_normalized_text_deduplication": len(frame),
        "duplicates_removed": duplicates,
        "design": "Five-fold GroupKFold by hotel. For each fold, train on one polarity from 16 hotels and test on either the same or opposite polarity from the four held-out hotels. Both labels are balanced within each polarity before deduplication.",
        "models": [
            "LogisticRegression(C=1,class_weight=balanced,max_iter=2000)",
            "LinearSVC(C=1,class_weight=balanced)",
            "MultinomialNB(alpha=0.5)",
        ],
        "text_pipeline": "Word TF-IDF, unigram and bigram, min_df=2, max_features=10000, sublinear_tf=True; fit on source-polarity training fold only.",
        "limitations": "Exploratory secondary analysis on the same controlled corpus; sentiment polarity is a coarse domain shift, not external-domain or deployment validation. Fold SD is descriptive and not a confidence interval.",
        "random_state": SEED,
    }
    (OUTPUT_DIR / "polarity_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
