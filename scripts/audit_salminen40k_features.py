"""Audit rating and text length as label-only shortcut baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "fake reviews dataset.csv"
OUTPUT_DIR = ROOT / "results" / "salminen40k"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    frame["label_id"] = (frame["label"] == "CG").astype(int)
    frame["normalized_text"] = (
        frame["text_"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    frame = frame.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    frame["log_word_count"] = np.log1p(frame["text_"].fillna("").astype(str).str.split().str.len())
    labels = frame["label_id"].to_numpy()
    groups = frame["category"].to_numpy()
    protocols = {
        "stratified_random_5fold": list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(frame, labels)
        ),
        "heldout_category_5fold": list(GroupKFold(n_splits=5).split(frame, labels, groups)),
    }
    rows = []
    for protocol, splits in protocols.items():
        for feature in ("rating", "log_word_count"):
            x = frame[[feature]].to_numpy()
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                )
                model.fit(x[train_idx], labels[train_idx])
                prediction = model.predict(x[test_idx])
                rows.append(
                    {
                        "protocol": protocol,
                        "feature": feature,
                        "fold": fold,
                        "test_categories": ",".join(sorted(set(groups[test_idx]))),
                        "accuracy": accuracy_score(labels[test_idx], prediction),
                        "macro_f1": f1_score(labels[test_idx], prediction, average="macro", zero_division=0),
                    }
                )
    folds = pd.DataFrame(rows)
    folds.to_csv(OUTPUT_DIR / "feature_only_fold_metrics.csv", index=False)
    summaries = []
    for (protocol, feature), group in folds.groupby(["protocol", "feature"], sort=False):
        summaries.append(
            {
                "protocol": protocol,
                "feature": feature,
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "feature_only_summary.csv", index=False, float_format="%.6f")
    summary.to_json(OUTPUT_DIR / "feature_only_metadata.json", orient="records", indent=2)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
