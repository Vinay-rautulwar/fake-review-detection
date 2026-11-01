# Unified Fake Review Detection - Model Evaluation & Comparative Benchmark

## 1. Overview & Evaluation Protocols

This unified evaluation compares all developed fake review detection models under strict leakage-isolated conditions:
- **Held-Out Test Set (E-Commerce)**: 4,474 reviews (2,797 Genuine, 1,677 Fake). 100% Reviewer ID and template cluster isolation. Scalers fit exclusively on `train.csv`. Optimal classification thresholds tuned exclusively on `validation.csv`.
- **Ott Deceptive Opinion Corpus**: 1,600 reviews (800 Truthful, 800 Deceptive, balanced across 20 hotels). 5-Fold Hotel-Disjoint Cross-Validation.

---

## 2. Comprehensive Model Comparison (Held-Out Test Set: 4,474 Reviews)

| Model Architecture | Modality | Threshold | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LightGBM (Default)** | Text-Only | 0.50 | 0.6450 | 0.4893 | 0.5340 | **0.5107** | 0.6691 | 0.6118 |
| **TF-IDF + LightGBM (Val-Tuned)** | Text-Only | 0.25 | 0.5029 | 0.4259 | 0.9433 | **0.5869** | 0.6691 | 0.6118 |
| **DeBERTa-v3 Text-Only (Default)** | Text-Only | 0.50 | 0.7143 | 0.8674 | 0.2809 | **0.4243** | 0.7152 | 0.6639 |
| **DeBERTa-v3 Text-Only (Val-Tuned)** | Text-Only | 0.12 | 0.5429 | 0.4468 | 0.9219 | **0.6019** | 0.7152 | 0.6639 |
| **Hybrid Fusion Net (Cross-Entropy)** | Multimodal (Text + Tabular) | 0.50 | 0.7101 | 0.6437 | 0.5075 | **0.5675** | 0.7461 | 0.6995 |
| **Hybrid Fusion Net (CE, Val-Tuned)** | Multimodal (Text + Tabular) | 0.26 | 0.5586 | 0.4551 | 0.9010 | **0.6048** | 0.7461 | 0.6995 |
| **Hybrid Fusion Net (Class-Weighted)** | Multimodal (Text + Tabular) | 0.50 | 0.6001 | 0.4803 | 0.8140 | **0.6041** | 0.7459 | 0.7000 |
| **Hybrid Fusion Net (Focal Loss, 0.50)** | Multimodal (Text + Tabular) | 0.50 | 0.6549 | 0.5316 | 0.6673 | **0.5918** | **0.7472** | **0.7002** |
| **Hybrid Fusion Net (Focal Loss, Val-Tuned)** | Multimodal (Text + Tabular) | 0.41 | 0.5300 | 0.4397 | 0.9267 | **0.5964** | **0.7472** | **0.7002** |
| **LightGBM Hybrid (Standard)** | Multimodal (Text + Tabular) | 0.50 | 0.7137 | 0.6911 | 0.4270 | **0.5278** | 0.7425 | 0.6953 |
| **LightGBM Hybrid (Balanced + PCA)** | Multimodal (Text + Tabular) | 0.50 | 0.6759 | 0.5625 | 0.6088 | **0.5848** | 0.7418 | 0.6923 |

---

## 3. Academic Controlled Corpus Comparison (Ott Hotel Deceptive Opinion)

| Model | Approach | Macro-F1 | Accuracy | Fake F1 | ROC-AUC | AP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Word TF-IDF)** | 5-Fold Hotel-Disjoint CV | **0.8859** (SD 0.013) | **0.8860** | **0.8841** | 0.9510 | 0.9530 |
| **BiLSTM (Packed Sequences)** | 5-Fold Hotel-Disjoint CV | **0.7526** (SD 0.037) | **0.7538** | **0.7512** | 0.8320 | 0.8390 |
| **Logistic Regression (TF-IDF)** | 5-Fold Hotel-Disjoint CV | **0.8714** (SD 0.014) | **0.8716** | **0.8700** | 0.9450 | 0.9470 |
| **Multinomial Naive Bayes** | 5-Fold Hotel-Disjoint CV | **0.8672** (SD 0.018) | **0.8672** | **0.8650** | 0.9410 | 0.9430 |
| **Multi-Layer Perceptron (MLP)** | 5-Fold Hotel-Disjoint CV | **0.8647** (SD 0.015) | **0.8647** | **0.8630** | 0.9380 | 0.9400 |

---

## 4. Key Scientific Findings & Hybrid Fusion Superiority

1. **Multimodal Fusion Delivers Decisive Lift**: Adding engineered behavioral metadata (rating discrepancy, reviewer history, verified purchase, reviewer velocity) improves ROC-AUC from **0.7152 to 0.7472** (+3.20 points) and PR-AUC from **0.6639 to 0.7002** (+3.63 points) over text-only DeBERTa.
2. **Focal Loss Resolves Extreme Recall Suppression**: Text-only DeBERTa at default 0.50 threshold suffers from severe recall collapse (recall: 0.2809) due to synthetic template noise and class imbalance. Hybrid Fusion with Focal Loss restores recall to **0.6673** at default threshold and **0.9267** when validation-tuned.
3. **Cross-Attention Explainability**: The `FakeGuard` multi-view self-attention mechanism produces a 3x3 inter-view attention matrix quantifying the dynamic relevance of linguistic, behavioral, and relational streams.
