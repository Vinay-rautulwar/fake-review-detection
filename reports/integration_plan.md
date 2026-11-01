# Repository Integration Plan: Combining Fake Review Detection with FakeGuard

## 1. Executive Summary

This document outlines the architectural comparison and integration plan to combine:
1. **Vinay Rautulwar's Project (`fake-review-detection-v2`)**: A complete, production-grade web service and API with a verified, leakage-isolated experimental pipeline, fine-tuned `microsoft/deberta-v3-base` transformer, multimodal DeBERTa + behavioral tabular fusion models, and modern web interface.
2. **Pramit Ghosh's Project (`FakeGuard`)**: A modular research framework implementing multi-view deep learning (Text, Behavior, Graph with GAT and Cross-Attention Fusion), reproducible 5-fold cross-validation benchmarks on the Ott hotel deception corpus, and comprehensive research auditing.

---

## 2. In-Depth Comparative Analysis

| Feature Area | Vinay's Project (`fake-review-detection-v2`) | Pramit's Project (`FakeGuard`) | Unified Integrated Architecture |
| :--- | :--- | :--- | :--- |
| **System Philosophy** | Full-Stack Production System & Verified Deployments | Modular Deep Learning Research Framework & Benchmarks | **Unified Platform**: Production REST API & UI integrated with modular research architecture (`src/`). |
| **Core Models** | - DeBERTa-v3-base (FP32 fine-tuned)<br>- TF-IDF + LightGBM baseline<br>- PyTorch DeBERTa + Tabular Hybrid Net<br>- LightGBM Hybrid (Balanced + PCA) | - Multi-View FakeGuard (Text + Behavior + GAT + Attention)<br>- Linear SVM (word/char TF-IDF)<br>- Packed-sequence BiLSTM<br>- Logistic Regression & Naive Bayes | **Unified Model Zoo**: Preserve all 8 distinct models and expose them for fair benchmarking and inference. |
| **Behavioral Stream** | 11 Engineered Tabular Features (discrepancy, verified ratio, review velocity, price) | Multi-Layer Perceptron (`BehaviorEncoder` with BatchNorm & Dropout) | Standardized behavioral feature vector shared between classical, GBDT, and neural encoders. |
| **Relational Stream** | Absent | Graph Attention Network (`GraphEncoder` with PyG & graceful fallback) | Integrated GAT module for modeling reviewer-product bipartite structures. |
| **Fusion Mechanism** | Concatenation + Deep MLP Head with Focal Loss | Multi-Head Self-Attention over projected view tokens + classification head | Both Fusion options supported: Cross-Attention Fusion with attention weights and Deep Tabular Fusion. |
| **Backend & Serving** | FastAPI with MongoDB, health check, prediction history, analytics, batch scoring | Prototype imported from prior commit | Upgraded FastAPI `ModelService` supporting `deberta_text`, `deberta_fusion`, `baseline`, and `fakeguard`. |
| **Frontend** | Modern glassmorphism UI with live analysis & radar charts | Prototype imported from prior commit | Polished web interface supporting all model types and attention visualizations. |
| **Evaluation Suite** | Test split benchmark (4,474 reviews) with threshold tuning | 5-Fold Grouped CV on Ott (1,600 reviews) & Salminen (40k) | **Unified Evaluation Suite**: Fair comparison across all datasets and models with identical metrics. |
| **Attribution** | Vinay Rautulwar | Pramit Ghosh | Co-attribution in documentation, papers, and package metadata. |

---

## 3. Integration Plan & Execution Steps

### Step 1: Core Research Library (`src/`) Integration
- Copy the modular package `src/` from friend's repo to root:
  - `src/models/`: `fakeguard.py`, `fusion.py`, `text_encoder.py`, `behavior_encoder.py`, `graph_encoder.py`.
  - `src/training/`: `losses.py` (Focal Loss, Label Smoothing), `trainer.py`.
  - `src/evaluation/`: `metrics.py` (`MetricsCalculator`), `explainability.py`.
  - `src/data/`: `dataset.py`, `feature_engineering.py`, `preprocessing.py`.
  - `src/utils/`: `helpers.py`.

### Step 2: Cross-Corpus Benchmark Scripts (`scripts/`) Integration
- Copy and adapt friend's benchmark scripts:
  - `scripts/run_ott_experiments.py` (Linear SVM, Logistic Regression, Naive Bayes, MLP).
  - `scripts/run_ott_bilstm.py` (Packed BiLSTM).
  - `scripts/tune_grouped_svm.py` (Nested CV tuning).
  - `scripts/run_salminen40k_experiments.py` & `run_salminen40k_mlp.py`.
  - `scripts/audit_ott_collection_bias.py` & `scripts/audit_salminen40k_features.py`.
  - `scripts/generate_figures.py`.

### Step 3: Experiment Configuration, Visuals & Research Audit
- Integrate `configs/config.yaml`.
- Integrate `results/`:
  - `results/figures/` (correlation heatmaps, distributions, wordclouds).
  - `results/presentation_diagrams/` (presentation visuals).
  - `results/model_tuning/` & `results/ott_experiments/`.
- Preserve and integrate `RESEARCH_AUDIT.md`.

### Step 4: Backend Model Service Upgrade (`backend/services/model_service.py`)
- Enhance `ModelService` to dynamically load:
  1. `deberta_text`: `backend/models/deberta_fp32_final`
  2. `deberta_fusion`: `backend/models/hybrid_deberta_tabular` (PyTorch Hybrid + Tabular Scaler)
  3. `baseline`: `training/artifacts/baseline` (TF-IDF + LightGBM)
  4. `fakeguard`: `src/models/fakeguard.py` (Multi-view cross-attention)
- Support returning attention weights when available.

### Step 5: Unified Model Evaluation Pipeline (`training/scripts/evaluate_unified_models.py`)
- Implement a single script to evaluate and compare all models on the held-out test split under identical leakage-isolated conditions.
- Produce `reports/unified_model_benchmark.json` and update `reports/model_comparison_benchmark.md`.

### Step 6: Test Suite Expansion
- Ensure all 29 backend tests pass.
- Create unit tests in `tests/test_model.py` that directly test `FakeGuard`, `CrossAttentionFusion`, `BehaviorEncoder`, `GraphEncoder`, and `FocalLoss`.

### Step 7: Packaging, Requirements & Documentation
- Merge `requirements.txt`, `requirements-backend.txt`, `requirements-training.txt`, and `setup.py`.
- Write `RESEARCH_PAPER_CHANGES.md`.
- Rewrite `README.md` with full attribution, architecture diagrams, benchmark tables, and reproduction steps.

### Step 8: Git Verification & Push
- Verify `git status` (respecting `.gitignore` for binary models and raw datasets).
- Commit with proper attribution to Vinay Rautulwar and Pramit Ghosh.
- Push to `https://github.com/Vinay-rautulwar/fake-review-detection.git`.
