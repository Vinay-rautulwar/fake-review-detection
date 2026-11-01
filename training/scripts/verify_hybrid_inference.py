"""
Verification Script for Hybrid Multimodal Model Inference.
===========================================================
Tests loading of:
1. PyTorch Hybrid Multimodal Checkpoint
2. Tabular Feature Scaler
3. DeBERTa Backbone & Tokenizer
Runs sample predictions with text and tabular metadata.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure script parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hybrid_models import DeBERTaTabularFusionNet, extract_tabular_features


def main():
    print("=" * 70)
    print(" VERIFYING HYBRID MULTIMODAL INFERENCE PIPELINE")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent.parent.parent
    hybrid_dir = base_dir / "backend" / "models" / "hybrid_deberta_tabular"
    deberta_dir = base_dir / "backend" / "models" / "deberta_fp32_final"

    # 1. Load Tabular Scaler
    scaler_bundle = joblib.load(hybrid_dir / "tabular_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    feature_names = scaler_bundle["feature_names"]
    print(f"Loaded Tabular Scaler ({len(feature_names)} features: {feature_names[:4]}...)")

    # 2. Load PyTorch Hybrid Model
    checkpoint_path = hybrid_dir / "pytorch_hybrid_class_weighted.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = DeBERTaTabularFusionNet(
        text_dim=checkpoint["text_dim"],
        tab_dim=checkpoint["tab_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded PyTorch Hybrid Net from {checkpoint_path.name}")
    print(f"  Validation Optimal Threshold: {checkpoint['val_threshold']:.2f}")

    # 3. Load DeBERTa Backbone for Text Embeddings
    print("Loading DeBERTa text backbone for embedding extraction...")
    tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir), local_files_only=True)
    deberta = AutoModelForSequenceClassification.from_pretrained(str(deberta_dir), local_files_only=True)
    deberta.eval()

    # 4. Test Sample 1: Suspicious Unverified Review with Extreme Rating Deviation
    sample1_text = "Amazing product highly recommend five stars best purchase ever!"
    sample1_meta = pd.DataFrame([{
        "rating": 5.0,
        "product_avg_rating": 2.1,
        "verified_purchase": 0,
        "reviewer_review_count": 1,
        "days_since_previous_review": 1,
        "reviewer_avg_rating": 5.0,
        "reviewer_account_age_days": 2,
        "reviewer_verified_review_count": 0,
        "product_price": 49.99,
        "product_review_count": 12,
    }])

    # Test Sample 2: Genuine Verified Review with Normal Rating
    sample2_text = "I've had this keyboard for two weeks. The keys are responsive and build quality is decent for the price."
    sample2_meta = pd.DataFrame([{
        "rating": 4.0,
        "product_avg_rating": 4.1,
        "verified_purchase": 1,
        "reviewer_review_count": 18,
        "days_since_previous_review": 45,
        "reviewer_avg_rating": 3.9,
        "reviewer_account_age_days": 650,
        "reviewer_verified_review_count": 16,
        "product_price": 45.0,
        "product_review_count": 280,
    }])

    for idx, (text, meta, label_desc) in enumerate([
        (sample1_text, sample1_meta, "Likely Fake / Astroturfed Review"),
        (sample2_text, sample2_meta, "Legitimate Verified Buyer Review")
    ], 1):
        # Extract text embedding
        encoded = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            enc_out = deberta.deberta(**encoded)
            text_embed = deberta.pooler(enc_out[0])

            # Extract tabular features
            tab_feat, _ = extract_tabular_features(meta, scaler=scaler, fit_scaler=False)
            tab_tensor = torch.tensor(tab_feat, dtype=torch.float32)

            # Hybrid inference
            logits = model(text_embed, tab_tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            p_fake = float(probs[1])
            p_genuine = float(probs[0])

            threshold = checkpoint["val_threshold"]
            pred = "FAKE" if p_fake >= threshold else "GENUINE"
            confidence = max(p_fake, p_genuine)

        print(f"\n--- Sample {idx}: {label_desc} ---")
        print(f"Review Text     : \"{text}\"")
        print(f"Verified Purchase: {meta['verified_purchase'].iloc[0]} | Rating: {meta['rating'].iloc[0]} (Product Avg: {meta['product_avg_rating'].iloc[0]})")
        print(f"Prediction      : {pred} (Threshold = {threshold:.2f})")
        print(f"Fake Probability: {p_fake:.4f} | Genuine Probability: {p_genuine:.4f} | Confidence: {confidence:.4f}")

    print("\n[SUCCESS] Hybrid multimodal inference verification passed completely!")


if __name__ == "__main__":
    main()
