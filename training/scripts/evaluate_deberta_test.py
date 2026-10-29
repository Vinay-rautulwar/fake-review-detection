"""
Evaluation Script for Trained DeBERTa-v3-base Model on Held-Out Test Set.

Evaluates the fine-tuned microsoft/deberta-v3-base model from backend/models/deberta_fp32_final
on data/splits/test.csv strictly in inference mode (torch.no_grad, model.eval()) without retraining
or tuning thresholds on the test data.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress PyTorch JIT warning on Python 3.14+
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("=" * 75)
    print(" FAKE REVIEW DETECTION - DEBERTA-V3 TEST SET EVALUATION")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent.parent.parent
    test_csv_path = base_dir / "data" / "splits" / "test.csv"
    model_dir = base_dir / "backend" / "models" / "deberta_fp32_final"

    # 1. Verify existence of test dataset and model files
    if not test_csv_path.exists():
        print(f"[ERROR] Test dataset not found at: {test_csv_path}")
        sys.exit(1)

    required_model_files = ["config.json", "model.safetensors"]
    missing_files = [f for f in required_model_files if not (model_dir / f).exists()]
    if missing_files:
        print(f"[ERROR] Missing model file(s) in {model_dir}: {missing_files}")
        sys.exit(1)

    print(f"Test Dataset Path : {test_csv_path}")
    print(f"Model Directory   : {model_dir}")

    # 2. Load test dataset
    df_test = pd.read_csv(test_csv_path)
    total_reviews = len(df_test)
    print(f"Total Test Reviews: {total_reviews:,}")

    # Validate columns
    if "review_text" not in df_test.columns or "label" not in df_test.columns:
        print("[ERROR] 'review_text' or 'label' column missing in test dataset.")
        sys.exit(1)

    # Standardize ground truth labels: 0 = genuine, 1 = fake
    label_map = {"genuine": 0, "fake": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    y_true = df_test["label"].map(label_map).values
    if np.isnan(y_true).any():
        print("[ERROR] Unrecognized values found in test labels.")
        sys.exit(1)
    y_true = y_true.astype(int)

    n_genuine = int((y_true == 0).sum())
    n_fake = int((y_true == 1).sum())
    print(f"Label Split       : Genuine (0): {n_genuine:,} ({n_genuine/total_reviews*100:.2f}%) | Fake (1): {n_fake:,} ({n_fake/total_reviews*100:.2f}%)")

    # 3. Configure hardware & threads
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Optimize CPU threads for multi-core batch processing
        threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"Inference Device  : CPU ({threads} PyTorch intra-op threads)")
    else:
        print(f"Inference Device  : GPU ({torch.cuda.get_device_name(0)})")

    # 4. Load model and tokenizer (local files only)
    print("\nLoading model & tokenizer (local_files_only=True)...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    model.to(device)
    print(f"Loaded successfully in {time.time() - load_start:.2f} seconds.")

    # 5. Length-sorted batching for maximum CPU throughput with dynamic padding
    texts = df_test["review_text"].astype(str).tolist()
    # Track original index so probabilities align perfectly with y_true
    indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
    sorted_orig_indices = [x[0] for x in indexed_texts]
    sorted_texts = [x[1] for x in indexed_texts]

    batch_size = 32
    all_fake_probs_sorted = []

    print(f"\nRunning test set inference in batches of {batch_size}...")
    eval_start = time.time()
    last_log_time = eval_start

    with torch.no_grad():
        for i in range(0, total_reviews, batch_size):
            batch_slice = sorted_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # Logits shape: [batch_size, 2]. Softmax over dim=-1, index 1 is fake probability
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_fake_probs_sorted.extend(probs)

            # Progress logging every ~20 seconds or 20%
            done = min(i + batch_size, total_reviews)
            now = time.time()
            if now - last_log_time >= 15 or done == total_reviews:
                pct = (done / total_reviews) * 100
                speed = done / (now - eval_start)
                print(f"  Processed {done:,}/{total_reviews:,} ({pct:5.1f}%) -- Speed: {speed:.1f} reviews/s")
                last_log_time = now

    total_time = time.time() - eval_start
    print(f"Inference completed in {total_time:.2f} seconds ({total_reviews/total_time:.1f} reviews/second).")

    # 6. Reconstruct probabilities in original dataframe order
    reconstructed_probs = np.zeros(total_reviews, dtype=float)
    for sorted_idx, orig_idx in enumerate(sorted_orig_indices):
        reconstructed_probs[orig_idx] = all_fake_probs_sorted[sorted_idx]

    # 7. Evaluate using default threshold = 0.50 (no test set tuning)
    threshold = 0.50
    y_pred = (reconstructed_probs >= threshold).astype(int)

    # 8. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, reconstructed_probs)
    pr_auc = average_precision_score(y_true, reconstructed_probs)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 9. Output formatted results
    print("\n" + "=" * 75)
    print(" OFFICIAL TEST SET BENCHMARK RESULTS (THRESHOLD = 0.50)")
    print("=" * 75)
    print(f"  Accuracy                : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision (Fake Class)  : {prec:.4f}  ({prec * 100:.2f}%)")
    print(f"  Recall (Fake Class)     : {rec:.4f}  ({rec * 100:.2f}%)")
    print(f"  F1-Score (Fake Class)   : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(f"  ROC-AUC Score           : {roc_auc:.4f}")
    print(f"  PR-AUC (Avg Precision)  : {pr_auc:.4f}")
    print("-" * 75)

    print("\nCONFUSION MATRIX (Actual vs. Predicted):")
    print("                     Predicted Genuine (0)    Predicted Fake (1)      Total Actual")
    print(f"  Actual Genuine (0)        {tn:6,d} (TN)              {fp:6,d} (FP)        {tn + fp:6,d}")
    print(f"  Actual Fake (1)           {fn:6,d} (FN)              {tp:6,d} (TP)        {fn + tp:6,d}")
    print(f"  Total Predicted           {tn + fn:6,d}                   {fp + tp:6,d}        {total_reviews:6,d}")

    print("\nDETAILED CLASSIFICATION REPORT:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["genuine (0)", "fake (1)"],
            digits=4,
        )
    )

    print("=" * 75)
    print("Evaluation finished successfully without retraining or test set tuning.")
    print("=" * 75)


if __name__ == "__main__":
    main()
