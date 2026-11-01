"""
Train and evaluate a tuned TF-IDF + Linear SVM classifier on the 17-column review dataset.
Adapted from Pramit Ghosh's group-aware tuned SVM pipeline (scripts/tune_grouped_svm.py).

Features:
- N-gram word TF-IDF (1, 2)
- Sublinear term frequency scaling
- Calibrated probability estimates for ROC-AUC / PR-AUC calculation
- Evaluation on validation and test sets (reviewer-held-out)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VAL_PATH = ROOT / "data" / "splits" / "validation.csv"
TEST_PATH = ROOT / "data" / "splits" / "test.csv"
ARTIFACTS_DIR = ROOT / "training" / "artifacts" / "svm"
REPORTS_DIR = ROOT / "reports"


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


def main():
    logger.info("Loading dataset splits...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    logger.info(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    def get_text_col(df: pd.DataFrame) -> list[str]:
        col = "review_text" if "review_text" in df.columns else "text"
        return df[col].fillna("").astype(str).tolist()

    def get_labels(df: pd.DataFrame) -> np.ndarray:
        series = df["label"].astype(str).str.lower().str.strip()
        return (series.isin(["fake", "1", "cg"])).values.astype(int)

    X_train_text = get_text_col(df_train)
    y_train = get_labels(df_train)

    X_val_text = get_text_col(df_val)
    y_val = get_labels(df_val)

    X_test_text = get_text_col(df_test)
    y_test = get_labels(df_test)

    logger.info("Fitting TF-IDF Vectorizer (ngram_range=(1, 2), max_features=15000)...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)
    X_test_vec = vectorizer.transform(X_test_text)

    logger.info("Training LinearSVC with Platt probability calibration...")
    start_time = time.time()
    base_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
    # CalibratedClassifierCV allows predict_proba
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    calibrated_svc.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    logger.info(f"SVM trained in {train_time:.2f} seconds.")

    # Validation evaluation
    val_probs = calibrated_svc.predict_proba(X_val_vec)[:, 1]
    val_metrics = compute_metrics(y_val, val_probs)
    logger.info(f"Validation Metrics: F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}")

    # Test evaluation
    test_probs = calibrated_svc.predict_proba(X_test_vec)[:, 1]
    test_metrics = compute_metrics(y_test, test_probs)
    logger.info(f"Test Metrics: F1={test_metrics['f1']:.4f}, Recall={test_metrics['recall']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}")

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(calibrated_svc, ARTIFACTS_DIR / "calibrated_svm.joblib")

    # Save test predictions for unified evaluation pipeline
    np.save(ARTIFACTS_DIR / "test_fake_probs.npy", test_probs)
    np.save(ARTIFACTS_DIR / "val_fake_probs.npy", val_probs)

    results = {
        "model": "TF-IDF + Linear SVM (Calibrated)",
        "ngram_range": [1, 2],
        "max_features": 15000,
        "class_weight": "balanced",
        "train_time_sec": train_time,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACTS_DIR / "svm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"SVM evaluation complete. Metrics saved to {ARTIFACTS_DIR / 'svm_metrics.json'}")


if __name__ == "__main__":
    main()
