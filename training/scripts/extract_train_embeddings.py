"""
Extract DeBERTa Embeddings and Predictions for Training Split.
==============================================================
Extracts and caches:
- train_fake_probs.npy
- train_embeddings.npy (shape: [20875, 768])
Using backend/models/deberta_fp32_final without modifying weights.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 75)
    print(" EXTRACTING DEBERTA EMBEDDINGS & PREDICTIONS FOR TRAIN SPLIT")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent.parent.parent
    train_csv_path = base_dir / "data" / "splits" / "train.csv"
    model_dir = base_dir / "backend" / "models" / "deberta_fp32_final"
    cache_dir = base_dir / "training" / "artifacts" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    probs_file = cache_dir / "train_fake_probs.npy"
    embeds_file = cache_dir / "train_embeddings.npy"

    if probs_file.exists() and embeds_file.exists():
        print(f"[CACHE HIT] Train embeddings and predictions already exist in {cache_dir}")
        embeds = np.load(embeds_file)
        probs = np.load(probs_file)
        print(f"Loaded embeddings shape: {embeds.shape}, probs shape: {probs.shape}")
        return

    df_train = pd.read_csv(train_csv_path)
    total = len(df_train)
    print(f"Total Train Reviews: {total:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"Device: CPU ({threads} PyTorch threads)")

    print(f"Loading DeBERTa model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    model.to(device)

    texts = df_train["review_text"].astype(str).tolist()
    indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
    sorted_orig_indices = [x[0] for x in indexed_texts]
    sorted_texts = [x[1] for x in indexed_texts]

    batch_size = 48
    all_probs = []
    all_embeddings = []

    print(f"\nRunning train set inference in batches of {batch_size}...")
    start_time = time.time()
    last_log = start_time

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_slice = sorted_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            encoder_outputs = model.deberta(**encoded)
            sequence_output = encoder_outputs[0]
            pooled_output = model.pooler(sequence_output)
            logits = model.classifier(model.dropout(pooled_output))

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            embeds = pooled_output.cpu().numpy()

            all_probs.extend(probs)
            all_embeddings.append(embeds)

            done = min(i + batch_size, total)
            now = time.time()
            if now - last_log >= 25 or done == total:
                pct = (done / total) * 100
                speed = done / (now - start_time)
                eta = (total - done) / speed if speed > 0 else 0
                print(f"  Processed {done:,}/{total:,} ({pct:5.1f}%) -- Speed: {speed:.1f} rev/s -- ETA: {eta:.0f}s")
                last_log = now

    all_embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.1f}s ({total/elapsed:.1f} reviews/s).")

    # Reconstruct original order
    reconstructed_probs = np.zeros(total, dtype=np.float32)
    reconstructed_embeddings = np.zeros((total, 768), dtype=np.float32)
    for sorted_idx, orig_idx in enumerate(sorted_orig_indices):
        reconstructed_probs[orig_idx] = all_probs[sorted_idx]
        reconstructed_embeddings[orig_idx] = all_embeddings[sorted_idx]

    np.save(probs_file, reconstructed_probs)
    np.save(embeds_file, reconstructed_embeddings)
    print(f"Saved: {probs_file} and {embeds_file}")


if __name__ == "__main__":
    main()
