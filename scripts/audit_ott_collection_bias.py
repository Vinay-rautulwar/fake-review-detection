"""Quantify provenance shortcuts and a review-length-only baseline on Ott data.

The source-only mapping is a validity diagnostic, not a permissible detection
model: it exposes that collection source and label are confounded in this corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "deceptive-opinion.csv"
OUTPUT_DIR = ROOT / "results" / "ott_experiments"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = (frame["deceptive"] == "deceptive").astype(int)
    frame["normalized_text"] = (
        frame["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    duplicates = int(frame.duplicated("normalized_text").sum())
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_table = (
        frame.groupby(["polarity", "source", "deceptive"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    source_table.to_csv(OUTPUT_DIR / "source_label_crosstab.csv", index=False)
    # More directly report whether each source maps to a unique label.
    label_counts_by_source = frame.groupby("source")["label"].nunique()
    perfect_source_mapping = bool(label_counts_by_source.eq(1).all())
    source_map = frame.groupby("source")["label"].agg(lambda values: values.mode().iat[0])
    mapped_labels = frame["source"].map(source_map).to_numpy()
    source_only_accuracy = accuracy_score(frame["label"], mapped_labels)
    source_only_macro_f1 = f1_score(frame["label"], mapped_labels, average="macro")

    frame["word_count"] = frame["text"].str.split().str.len()
    frame["log_word_count"] = np.log1p(frame["word_count"])
    features = frame[["log_word_count"]].to_numpy()
    labels = frame["label"].to_numpy()
    groups = frame["hotel"].to_numpy()
    records = []
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=5).split(features, labels, groups), start=1
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        )
        model.fit(features[train_idx], labels[train_idx])
        predictions = model.predict(features[test_idx])
        records.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_hotels": ",".join(sorted(set(groups[test_idx]))),
                "accuracy": accuracy_score(labels[test_idx], predictions),
                "macro_f1": f1_score(labels[test_idx], predictions, average="macro", zero_division=0),
            }
        )
    folds = pd.DataFrame(records)
    folds.to_csv(OUTPUT_DIR / "length_only_fold_metrics.csv", index=False)
    summary = {
        "dataset_sha256": sha256_file(DATA_PATH),
        "raw_rows": 1600,
        "rows_after_exact_normalized_text_deduplication": len(frame),
        "duplicates_removed": duplicates,
        "source_label_counts": source_table.to_dict(orient="records"),
        "every_source_maps_to_only_one_label": perfect_source_mapping,
        "source_only_diagnostic_accuracy": source_only_accuracy,
        "source_only_diagnostic_macro_f1": source_only_macro_f1,
        "length_only_model": "one feature log(1 + whitespace word count); StandardScaler and logistic regression fit on each training fold",
        "length_only_protocol": "five-fold GroupKFold by hotel",
        "length_only_accuracy_mean": float(folds["accuracy"].mean()),
        "length_only_accuracy_std": float(folds["accuracy"].std(ddof=1)),
        "length_only_macro_f1_mean": float(folds["macro_f1"].mean()),
        "length_only_macro_f1_std": float(folds["macro_f1"].std(ddof=1)),
        "interpretation": "The source-only number is an intentionally invalid shortcut diagnostic, not a valid model result. Length-only scores are a simple artifact baseline. Neither supports real-world fraud claims.",
        "scikit_learn": sklearn.__version__,
    }
    (OUTPUT_DIR / "collection_bias_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Length-only folds:")
    print(folds.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
