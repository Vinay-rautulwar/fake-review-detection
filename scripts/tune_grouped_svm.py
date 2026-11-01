"""Nested group-aware tuning for the leading TF-IDF SVM on both corpora.

The outer GroupKFold is reserved for evaluation. Within each outer training
fold, a separate GroupKFold selects SVM C, class weighting, and word n-gram
range by macro-F1. This prevents hyperparameter selection from using the
outer test groups.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "model_tuning"
SEED = 42
C_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
NGRAM_RANGES = ((1, 1), (1, 2))
OUTER_FOLDS = 5
INNER_FOLDS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ott() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "data" / "datasets" / "deceptive-opinion.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["deceptive"].isin(["deceptive", "truthful"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = (frame["deceptive"] == "deceptive").astype(int)
    frame["normalized_text"] = (
        frame["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    if frame.groupby("normalized_text")["label"].nunique().gt(1).any():
        raise ValueError("Ott corpus has normalized duplicate texts with conflicting labels")
    duplicates = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    return frame, {
        "name": "Ott deceptive-opinion hotel reviews",
        "path": path,
        "sha256": sha256_file(path),
        "text_column": "text",
        "group_column": "hotel",
        "positive_label": "deceptive",
        "duplicates_removed": duplicates,
        "max_features": 10000,
    }


def load_salminen() -> tuple[pd.DataFrame, dict]:
    path = ROOT / "data" / "datasets" / "fake reviews dataset.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["label"].isin(["CG", "OR"])].copy()
    frame["text_"] = frame["text_"].fillna("").astype(str)
    frame["label_id"] = (frame["label"] == "CG").astype(int)
    frame["normalized_text"] = (
        frame["text_"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    if frame.groupby("normalized_text")["label_id"].nunique().gt(1).any():
        raise ValueError("Product corpus has normalized duplicate texts with conflicting labels")
    duplicates = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    return frame, {
        "name": "Salminen-derived CG versus OR product reviews",
        "path": path,
        "sha256": sha256_file(path),
        "text_column": "text_",
        "group_column": "category",
        "positive_label": "CG",
        "duplicates_removed": duplicates,
        "max_features": 30000,
    }


def vectorizer(max_features: int, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )


def classifier(c_value: float, class_weight: str | None) -> LinearSVC:
    return LinearSVC(C=c_value, class_weight=class_weight, random_state=SEED)


def metric_record(y_true: np.ndarray, prediction: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "positive_f1": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def tune_corpus(frame: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = frame[spec["text_column"]]
    labels = frame["label" if spec["positive_label"] == "deceptive" else "label_id"].to_numpy()
    groups = frame[spec["group_column"]].to_numpy()
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    outer_records = []
    search_records = []

    for outer_fold, (outer_train, outer_test) in enumerate(
        outer.split(texts, labels, groups), start=1
    ):
        print(f"{spec['name']}: outer fold {outer_fold}/{OUTER_FOLDS}", flush=True)
        train_texts = texts.iloc[outer_train].reset_index(drop=True)
        train_labels = labels[outer_train]
        train_groups = groups[outer_train]
        test_texts = texts.iloc[outer_test]
        test_labels = labels[outer_test]
        inner_splits = list(
            GroupKFold(n_splits=INNER_FOLDS).split(train_texts, train_labels, train_groups)
        )

        candidate_scores: dict[tuple, list[float]] = {
            (ngram, c_value, weight): []
            for ngram, c_value, weight in itertools.product(
                NGRAM_RANGES, C_VALUES, CLASS_WEIGHTS
            )
        }

        for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
            x_train_text = train_texts.iloc[inner_train]
            x_valid_text = train_texts.iloc[inner_valid]
            y_train = train_labels[inner_train]
            y_valid = train_labels[inner_valid]
            for ngram in NGRAM_RANGES:
                tfidf = vectorizer(spec["max_features"], ngram)
                x_train = tfidf.fit_transform(x_train_text)
                x_valid = tfidf.transform(x_valid_text)
                for c_value, weight in itertools.product(C_VALUES, CLASS_WEIGHTS):
                    model = classifier(c_value, weight)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_valid)
                    key = (ngram, c_value, weight)
                    candidate_scores[key].append(
                        float(f1_score(y_valid, prediction, average="macro", zero_division=0))
                    )
            print(
                f"  inner fold {inner_fold}/{INNER_FOLDS} complete",
                flush=True,
            )

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (
                -float(np.mean(item[1])),
                item[0][1] != 1.0,
                item[0][0] != (1, 2),
                item[0][2] != "balanced",
            ),
        )
        (best_ngram, best_c, best_weight), best_inner_scores = ranked[0]
        search_records.append(
            {
                "outer_fold": outer_fold,
                "best_inner_macro_f1_mean": float(np.mean(best_inner_scores)),
                "best_inner_macro_f1_std": float(np.std(best_inner_scores, ddof=1)),
                "selected_ngram_range": f"{best_ngram[0]}-{best_ngram[1]}",
                "selected_C": best_c,
                "selected_class_weight": best_weight or "none",
                "candidates_evaluated": len(candidate_scores),
            }
        )

        final_vectorizer = vectorizer(spec["max_features"], best_ngram)
        x_train_final = final_vectorizer.fit_transform(texts.iloc[outer_train])
        x_test_final = final_vectorizer.transform(test_texts)
        final_model = classifier(best_c, best_weight)
        final_model.fit(x_train_final, labels[outer_train])
        prediction = final_model.predict(x_test_final)
        scores = final_model.decision_function(x_test_final)
        test_groups = sorted(set(groups[outer_test]))
        outer_records.append(
            {
                "fold": outer_fold,
                "n_train": len(outer_train),
                "n_test": len(outer_test),
                "test_groups": ",".join(map(str, test_groups)),
                "selected_ngram_range": f"{best_ngram[0]}-{best_ngram[1]}",
                "selected_C": best_c,
                "selected_class_weight": best_weight or "none",
                **metric_record(labels[outer_test], prediction, scores),
            }
        )

    return pd.DataFrame(outer_records), pd.DataFrame(search_records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for loader in (load_ott, load_salminen):
        frame, spec = loader()
        fold_metrics, selected = tune_corpus(frame, spec)
        stem = "ott" if spec["positive_label"] == "deceptive" else "salminen40k"
        fold_metrics.to_csv(OUTPUT_DIR / f"{stem}_nested_svm_folds.csv", index=False)
        selected.to_csv(OUTPUT_DIR / f"{stem}_nested_svm_selected_params.csv", index=False)
        summary = {
            "dataset": spec["name"],
            "protocol": "nested_group_5_outer_x_3_inner",
            "model": "LinearSVC; inner selection by macro-F1",
            "positive_label": spec["positive_label"],
            "folds": len(fold_metrics),
            "duplicates_removed": spec["duplicates_removed"],
        }
        for metric in (
            "accuracy",
            "macro_f1",
            "positive_precision",
            "positive_recall",
            "positive_f1",
            "roc_auc",
            "average_precision",
        ):
            summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
            summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=1))
        pd.DataFrame([summary]).to_csv(
            OUTPUT_DIR / f"{stem}_nested_svm_summary.csv", index=False, float_format="%.6f"
        )
        specs.append(
            {
                **{key: value for key, value in spec.items() if key != "path"},
                "path": str(spec["path"].relative_to(ROOT)),
                "scikit_learn": sklearn.__version__,
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "C_values": C_VALUES,
                "class_weights": CLASS_WEIGHTS,
                "ngram_ranges": NGRAM_RANGES,
                "vectorizer": {
                    "min_df": 2,
                    "max_features": spec["max_features"],
                    "sublinear_tf": True,
                },
                "selection_metric": "macro_f1",
                "warning": "Scores are still benchmark-specific; grouped nesting does not remove dataset source-label confounds or change label meaning.",
            }
        )

    (OUTPUT_DIR / "nested_tuning_metadata.json").write_text(
        json.dumps(specs, indent=2), encoding="utf-8"
    )
    for spec in specs:
        stem = "ott" if spec["positive_label"] == "deceptive" else "salminen40k"
        print(f"\n{stem} tuned outer-fold summary:")
        print(pd.read_csv(OUTPUT_DIR / f"{stem}_nested_svm_summary.csv").to_string(index=False))
        print(pd.read_csv(OUTPUT_DIR / f"{stem}_nested_svm_selected_params.csv").to_string(index=False))


if __name__ == "__main__":
    main()
