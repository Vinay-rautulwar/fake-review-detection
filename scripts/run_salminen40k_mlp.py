"""Category-held-out TF-IDF MLP on the GPT-2-vs-Amazon-original task."""

from __future__ import annotations

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datasets" / "fake reviews dataset.csv"
OUTPUT_DIR = ROOT / "results" / "salminen40k"
SEED = 42


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
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
    labels = frame["label_id"].to_numpy()
    texts = frame["text_"]
    categories = frame["category"].to_numpy()
    splits = GroupKFold(n_splits=5).split(texts, labels, categories)
    records = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"Training category-held-out MLP, fold {fold}/5", flush=True)
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=30000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=50,
                        early_stopping=True,
                        n_iter_no_change=5,
                        validation_fraction=0.1,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        pipeline.fit(texts.iloc[train_idx], y_train)
        prediction = pipeline.predict(texts.iloc[test_idx])
        probabilities = pipeline.predict_proba(texts.iloc[test_idx])[:, list(pipeline.classes_).index(1)]
        records.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_categories": ",".join(sorted(set(categories[test_idx]))),
                "best_iteration": pipeline.named_steps["classifier"].n_iter_,
                "accuracy": accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                "cg_precision": precision_score(y_test, prediction, pos_label=1, zero_division=0),
                "cg_recall": recall_score(y_test, prediction, pos_label=1, zero_division=0),
                "cg_f1": f1_score(y_test, prediction, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(y_test, probabilities),
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "mlp_fold_metrics.csv", index=False)
    metrics = ["accuracy", "macro_f1", "cg_precision", "cg_recall", "cg_f1", "roc_auc", "average_precision"]
    summary = {"protocol": "heldout_category_5fold", "model": "TF-IDF MLP (64 hidden units)", "folds": len(result), "duplicates_removed": duplicates}
    for metric in metrics:
        summary[f"{metric}_mean"] = float(result[metric].mean())
        summary[f"{metric}_std"] = float(result[metric].std(ddof=1))
    summary["best_iteration_mean"] = float(result["best_iteration"].mean())
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "mlp_summary.csv", index=False, float_format="%.6f")
    (OUTPUT_DIR / "mlp_metadata.json").write_text(
        json.dumps(
            {
                **summary,
                "dataset_sha256": __import__("hashlib").sha256(DATA_PATH.read_bytes()).hexdigest(),
                "scikit_learn": sklearn.__version__,
                "interpretation_limit": "This is an MLP on TF-IDF, not a pretrained Transformer or the full multi-view model. Labels are GPT-2-generated vs Amazon-original, not verified maliciousness.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
