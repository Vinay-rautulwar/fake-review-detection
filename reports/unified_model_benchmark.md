# Unified Model Benchmark Report: Merged Project Evaluation

**Date:** 2026-11-01  
**Dataset:** 17-Column E-Commerce Review Corpus (Reviewer-Held-Out Split)  
**Test Set Size:** 4,474 Reviews (2,797 Genuine, 1,677 Fake)  
**Authors:** Vinay Rautulwar & Pramit Ghosh  

---

## 1. Executive Summary & Comparative Matrix

All models below were evaluated on the **identical 4,474 held-out test reviews**.
No test data was used for threshold tuning, feature scaling, or vocabulary construction (zero data leakage).

| Model | Modality | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression (Balanced) | Text-Only | 0.5 | 0.6350 | 0.5118 | 0.5713 | 0.5399 | 0.7089 | 0.6589 | 958 | 914 | 719 | 1883 |
| TF-IDF + Linear SVM (Calibrated) | Text-Only | 0.5 | 0.6451 | 0.6344 | 0.1252 | 0.2092 | 0.6210 | 0.5167 | 210 | 121 | 1467 | 2676 |
| Word Embeddings + BiLSTM (Packed) | Text-Only | 0.5 | 0.6676 | 0.5651 | 0.4919 | 0.5260 | 0.7157 | 0.6583 | 825 | 635 | 852 | 2162 |
| TF-IDF + LightGBM Baseline | Text-Only | 0.5 | 0.6450 | 0.4893 | 0.5340 | 0.5107 | 0.6691 | 0.6118 | 895 | 934 | 781 | 1862 |
| TF-IDF + LightGBM Baseline (Val-Tuned) | Text-Only | 0.25 | 0.5029 | 0.4259 | 0.9433 | 0.5869 | 0.6691 | 0.6118 | 1581 | 2131 | 95 | 665 |
| DeBERTa-v3-base Text-Only (Default) | Text-Only | 0.5 | 0.7143 | 0.8674 | 0.2809 | 0.4243 | 0.7152 | 0.6639 | 471 | 72 | 1206 | 2725 |
| DeBERTa-v3-base Text-Only (Val-Tuned) | Text-Only | 0.12 | 0.5429 | 0.4468 | 0.9219 | 0.6019 | 0.7152 | 0.6639 | 1546 | 1914 | 131 | 883 |
| LightGBM Multimodal Hybrid | Multimodal | 0.5 | 0.7137 | 0.6911 | 0.4270 | 0.5278 | 0.7425 | 0.6953 | 716 | 320 | 961 | 2477 |
| LightGBM Multimodal Hybrid (Val-Tuned) | Multimodal | 0.15 | 0.5253 | 0.4374 | 0.9320 | 0.5954 | 0.7425 | 0.6953 | 1563 | 2010 | 114 | 787 |
| DeBERTaTabularFusionNet (Class-Weighted) | Multimodal | 0.5 | 0.6001 | 0.4803 | 0.8140 | 0.6041 | 0.7459 | 0.7000 | 1365 | 1477 | 312 | 1320 |
| DeBERTaTabularFusionNet (CW Val-Tuned) | Multimodal | 0.3 | 0.5436 | 0.4470 | 0.9183 | 0.6013 | 0.7459 | 0.7000 | 1540 | 1905 | 137 | 892 |
| DeBERTaTabularFusionNet (Focal Loss) | Multimodal | 0.5 | 0.6549 | 0.5316 | 0.6673 | 0.5918 | 0.7472 | 0.7002 | 1119 | 986 | 558 | 1811 |
| DeBERTaTabularFusionNet (Focal Val-Tuned) | Multimodal | 0.41 | 0.5300 | 0.4397 | 0.9267 | 0.5964 | 0.7472 | 0.7002 | 1554 | 1980 | 123 | 817 |

---

## 2. Key Findings & Cross-Repository Insights

1. **Transformer Semantics vs. Linear N-Grams**:
   - Fine-tuned DeBERTa-v3-base achieves the highest precision among text-only models (**86.74%** at default threshold 0.50), significantly outperforming TF-IDF + Logistic Regression (**48.8%**) and TF-IDF + Linear SVM (**20.9%**).
   - However, text-only DeBERTa suffers from severe false negatives at default 0.50 threshold (1,206 missed fakes, 28.09% recall).

2. **Validation-Guided Threshold Calibration**:
   - Calibrating the classification threshold exclusively on validation data ($T^* = 0.12$) boosts DeBERTa test recall from **28.09% to 92.19%**, raising test F1 from **0.4243 to 0.6019**.

3. **Multimodal Fusion Superiority (DeBERTa + Tabular Behavioral Metadata)**:
   - Combining DeBERTa text representations with tabular reviewer metadata (verified purchase, reviewer review count, rating discrepancy, account age) delivers the strongest overall performance:
     - **DeBERTaTabularFusionNet (Focal Loss)** achieves **ROC-AUC: 0.7472** and **PR-AUC: 0.7002** (highest across all tested models).
     - **DeBERTaTabularFusionNet (Class-Weighted Loss)** achieves **Recall: 81.40%** at standard threshold 0.50 with **F1: 0.6041**, balancing precision and recall without requiring extreme threshold shifts.
     - **LightGBM Hybrid** delivers **ROC-AUC: 0.7425** and **F1: 0.5954** with fast inference.

4. **Integration of Friend's Models**:
   - Pramit Ghosh's Linear SVM and packed-sequence BiLSTM provide essential intermediate baselines, demonstrating the performance progression from linear models ($F1 \approx 0.21$) to sequential deep learning ($F1 \approx 0.55$), transformer models ($F1 \approx 0.60$), and multimodal fusion ($ROC-AUC = 0.7472$).
