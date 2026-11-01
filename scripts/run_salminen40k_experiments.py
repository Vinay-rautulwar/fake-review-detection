"""Evaluate the GPT-2-generated vs Amazon-original review corpus.

Task labels are generation source (CG vs OR), not verified maliciousness.
The category-held-out protocol tests transfer to unseen Amazon product
categories. Exact text duplicates are removed before folds are created.

Run from the project root with: python scripts/run_salminen40k_experiments.py
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "fake reviews dataset.csv"
OUTPUT_DIR = ROOT / "results" / "salminen40k"
SEED = 42
METRICS = (
    "accuracy",
    "macro_f1",
    "cg_precision",
    "cg_recall",
    "cg_f1",
    "roc_auc",
    "average_precision",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(DATA_PATH)
    required = {"category", "rating", "label", "text_"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[frame["label"].isin(["CG", "OR"])].copy()
    frame["text_"] = frame["text_"].fillna("").astype(str)
    frame["label_id"] = (frame["label"] == "CG").astype(int)
    frame["normalized_text"] = (
        frame["text_"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    if frame.groupby("normalized_text")["label_id"].nunique().gt(1).any():
        raise ValueError("Exact normalized duplicate texts have conflicting labels")
    duplicates = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    return frame, duplicates


def models() -> dict[str, Pipeline]:
    vectorizer = {
        "max_features": 30000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    return {
        "Logistic Regression (word TF-IDF)": Pipeline(
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
        "Linear SVM (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer)),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                ),
            ]
        ),
        "Multinomial NB (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer)),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def score_vector(model: Pipeline, texts: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, list(model.classes_).index(1)]
    return np.asarray(model.decision_function(texts))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cg_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "cg_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "cg_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
    }


def main() -> None:
    frame, duplicate_count = load_data()
    texts = frame["text_"]
    labels = frame["label_id"].to_numpy()
    categories = frame["category"].to_numpy()
    protocols = {
        "stratified_random_5fold": list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(texts, labels)
        ),
        "heldout_category_5fold": list(
            GroupKFold(n_splits=5).split(texts, labels, categories)
        ),
    }
    records: list[dict] = []
    split_rows: list[dict] = []
    templates = models()
    for protocol, splits in protocols.items():
        for model_name, template in templates.items():
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                print(f"Fitting {protocol} | {model_name} | fold {fold}/5", flush=True)
                model = sklearn.base.clone(template)
                model.fit(texts.iloc[train_idx], labels[train_idx])
                predictions = model.predict(texts.iloc[test_idx])
                scores = score_vector(model, texts.iloc[test_idx])
                test_categories = sorted(set(categories[test_idx]))
                records.append(
                    {
                        "protocol": protocol,
                        "model": model_name,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_cg_test": int(labels[test_idx].sum()),
                        "test_categories": ",".join(test_categories),
                        **calculate_metrics(labels[test_idx], predictions, scores),
                    }
                )
                for idx in train_idx:
                    split_rows.append(
                        {"protocol": protocol, "fold": fold, "row_index": int(idx), "role": "train", "category": categories[idx], "label": int(labels[idx])}
                    )
                for idx in test_idx:
                    split_rows.append(
                        {"protocol": protocol, "fold": fold, "row_index": int(idx), "role": "test", "category": categories[idx], "label": int(labels[idx])}
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(records)
    fold_frame.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).drop_duplicates().to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
    summary_rows = []
    for (protocol, model_name), group in fold_frame.groupby(["protocol", "model"], sort=False):
        result = {"protocol": protocol, "model": model_name, "folds": len(group)}
        for metric in METRICS:
            result[f"{metric}_mean"] = group[metric].mean()
            result[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, float_format="%.6f")

    crosstab = pd.crosstab(frame["category"], frame["label"], margins=True)
    crosstab.to_csv(OUTPUT_DIR / "category_label_counts.csv")
    lengths = frame.assign(word_count=frame["text_"].str.split().str.len())
    length_summary = lengths.groupby("label")["word_count"].agg(["count", "mean", "std", "median"])
    length_summary.to_csv(OUTPUT_DIR / "text_length_by_label.csv")
    duplicate_groups = frame.groupby("normalized_text")["label_id"].nunique()
    prefix5 = frame["text_"].str.casefold().str.split().str[:5].str.join(" ")
    prefix5_by_class = {
        label: set(prefix5[frame["label"] == label]) for label in ("CG", "OR")
    }
    metadata = {
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATA_PATH),
        "primary_source_reference": "Salminen et al., Journal of Retailing and Consumer Services 64 (2022), article 102771, DOI 10.1016/j.jretconser.2021.102771",
        "primary_source_url": "https://doi.org/10.1016/j.jretconser.2021.102771",
        "task": "GPT-2 computer-generated review (CG) versus original Amazon review (OR); not a verified human-fraud label",
        "source_article_reports": "40,000 records (20,000 generated and 20,000 original); generated reviews use a fine-tuned GPT-2 continuation from the first five words and are length-matched by target buckets.",
        "local_mirror_rows_before_deduplication": int(pd.read_csv(DATA_PATH).shape[0]),
        "rows_after_exact_normalized_text_deduplication": len(frame),
        "exact_duplicates_removed": duplicate_count,
        "cross_label_exact_duplicate_groups": int(duplicate_groups.gt(1).sum()),
        "label_counts_after_deduplication": frame["label"].value_counts().to_dict(),
        "category_count": int(frame["category"].nunique()),
        "category_counts_equal_between_labels": bool(
            (pd.crosstab(frame["category"], frame["label"])["CG"]
             == pd.crosstab(frame["category"], frame["label"])["OR"]).all()
        ),
        "five_token_prefix_overlap_unique_rate_cg": len(prefix5_by_class["CG"] & prefix5_by_class["OR"]) / max(1, len(prefix5_by_class["CG"])),
        "protocols": {
            "stratified_random_5fold": "StratifiedKFold(shuffle=True, random_state=42); category overlap is allowed",
            "heldout_category_5fold": "GroupKFold(5) by product category; test folds contain two categories absent from training",
        },
        "text_pipeline": "word TF-IDF, unigram+bigrams, min_df=2, max_features=30000, sublinear_tf=True; fit within training fold",
        "models": [
            "LogisticRegression(C=1,class_weight=balanced,max_iter=2000)",
            "LinearSVC(C=1,class_weight=balanced)",
            "MultinomialNB(alpha=0.5)",
        ],
        "metrics": list(METRICS),
        "scikit_learn": sklearn.__version__,
        "interpretation_limit": "The benchmark measures GPT-2-vs-Amazon-original source discrimination. OR reviews are presumed human-authored and may contain unknown fraud; this is not a direct test of human-written deceptive-review detection or current-generation LLMs.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Category x label counts:")
    print(crosstab.to_string())
    print("Text length by label:")
    print(length_summary.to_string(float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
