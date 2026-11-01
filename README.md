# Fake Review Detection: Multimodal & Multi-View Deception Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-DeBERTa--v3-yellow.svg)](https://huggingface.co/microsoft/deberta-v3-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning system engineered to detect fraudulent and deceptive e-commerce product reviews by synthesizing contextual transformer embeddings with reviewer behavioral metadata, relational signals, and multi-view attention.

This unified repository integrates the complementary methodologies developed by **Vinay Rautulwar** and **Pramit Ghosh**.

---

## 👥 Contributors & Attribution

| Contributor | Focus Areas & Key Contributions | Original Repository |
| :--- | :--- | :--- |
| **Vinay Rautulwar** | • 17-column e-commerce review cleaning and validation pipeline<br>• Leakage-free 70/15/15 reviewer-held-out data partitioning<br>• Fine-tuned DeBERTa-v3-base transformer model<br>• Validation-guided threshold calibration ($T^* = 0.12$)<br>• Multimodal fusion (`DeBERTaTabularFusionNet`, LightGBM Hybrid)<br>• Production FastAPI REST service, MongoDB repository, health checks | [Vinay-rautulwar/fake-review-detection](https://github.com/Vinay-rautulwar/fake-review-detection.git) |
| **Pramit Ghosh** | • Multi-view `FakeGuard` framework (Text, Behavior, GAT Graph)<br>• Cross-attention view fusion module (`CrossAttentionFusion`)<br>• Packed-sequence BiLSTM and tuned Linear SVM text models<br>• Ott Deceptive Opinion Corpus and Salminen 40K benchmarking<br>• Empirical research audit (`RESEARCH_AUDIT.md`) and bias diagnostics<br>• Express.js API Gateway microservice | [PramitGhosh02/Fake-Review-Detection](https://github.com/PramitGhosh02/Fake-Review-Detection.git) |

---

## 🏗️ System Architecture

```text
                                 +--------------------------------+
                                 |     Customer Review Input      |
                                 |  Text + 13 Behavioral Features |
                                 +--------------------------------+
                                                 |
                   +-----------------------------+-----------------------------+
                   |                                                           |
                   v                                                           v
+--------------------------------------+                   +---------------------------------------+
|        Semantic View Encoders        |                   |      Behavioral & Relational Views    |
| • TF-IDF (1,2-grams)                 |                   | • StandardScaler Normalizer           |
| • BiLSTM (Packed Word Embeddings)    |                   | • MLP Behavior Encoder (128-d)        |
| • DeBERTa-v3-base Transformer (768-d)|                   | • GAT Graph Encoder (PyG / Fallback)  |
+--------------------------------------+                   +---------------------------------------+
                   |                                                           |
                   +-----------------------------+-----------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |              Fusion & Decision Heads              |
                       |                                                   |
                       | 1. TF-IDF + LightGBM Baseline                     |
                       | 2. Calibrated Linear SVM & Packed BiLSTM          |
                       | 3. DeBERTa-v3 Text-Only (T*=0.12)                 |
                       | 4. LightGBM Multimodal Hybrid                     |
                       | 5. DeBERTaTabularFusionNet (CW & Focal Loss)      |
                       | 6. FakeGuard Cross-Attention Multi-View Fusion   |
                       +---------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |        Calibrated Prediction & Explanation        |
                       |  p(Fake), Label, Confidence, Cross-View Attention |
                       +---------------------------------------------------+
```

---

## 📊 Unified Model Benchmark (Identical Test Split)

All models evaluated on the **identical 4,474 held-out reviewer test reviews** (2,797 Genuine, 1,677 Fake). No test instances were used for threshold selection, vocabulary creation, or scaling.

| Model | Modality | Threshold | Accuracy | Precision | Recall | Fake $F_1$ | ROC-AUC | PR-AUC (AP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LightGBM Baseline** | Text-Only | 0.50 | 0.6450 | 0.4893 | 0.5340 | 0.5107 | 0.6691 | 0.6118 |
| **TF-IDF + LightGBM (Val-Tuned)** | Text-Only | 0.25 | 0.5029 | 0.4259 | **0.9433** | 0.5869 | 0.6691 | 0.6118 |
| **DeBERTa-v3 Text-Only (Default)** | Text-Only | 0.50 | **0.7143** | **0.8674** | 0.2809 | 0.4243 | 0.7152 | 0.6639 |
| **DeBERTa-v3 Text-Only (Val-Tuned)**| Text-Only | 0.12 | 0.5429 | 0.4468 | 0.9219 | 0.6019 | 0.7152 | 0.6639 |
| **LightGBM Multimodal Hybrid** | Multimodal | 0.50 | 0.7137 | 0.6911 | 0.4270 | 0.5278 | 0.7425 | 0.6953 |
| **LightGBM Hybrid (Balanced + PCA)** | Multimodal | 0.50 | 0.6759 | 0.5625 | 0.6088 | 0.5848 | 0.7418 | 0.6923 |
| **DeBERTa FusionNet (Class-Weighted)** | Multimodal | 0.50 | 0.6001 | 0.4803 | **0.8140** | **0.6041** | 0.7459 | 0.7000 |
| **DeBERTa FusionNet (Focal Loss, 0.50)** | Multimodal | 0.50 | 0.6549 | 0.5316 | 0.6673 | 0.5918 | **0.7472** | **0.7002** |
| **DeBERTa FusionNet (Focal Val-Tuned)** | Multimodal | 0.41 | 0.5300 | 0.4397 | 0.9267 | 0.5964 | **0.7472** | **0.7002** |

### Controlled Corpus Benchmark (Ott Deceptive Opinion Hotel Reviews, 5-Fold Grouped CV):
- **Linear SVM (Word TF-IDF)**: Macro-F1 **0.8859** ± 0.013 | Accuracy: **0.8860** | ROC-AUC: 0.9510 | AP: 0.9530
- **Logistic Regression (TF-IDF)**: Macro-F1 0.8714 ± 0.014 | Accuracy: 0.8716 | ROC-AUC: 0.9450 | AP: 0.9470
- **Multinomial Naive Bayes**: Macro-F1 0.8672 ± 0.018 | Accuracy: 0.8672 | ROC-AUC: 0.9410 | AP: 0.9430
- **Packed-Sequence BiLSTM**: Macro-F1 0.7526 ± 0.037 | Accuracy: 0.7538 | ROC-AUC: 0.8320 | AP: 0.8390

---

## 📁 Repository Structure

```text
fake-review-detection/
├── backend/
│   ├── main.py                         # FastAPI application entrypoint & lifecycle
│   ├── api/routes/                     # Modular endpoints (health, predictions, models, analytics)
│   ├── core/config.py                  # Pydantic Settings configuration
│   ├── db/mongodb.py                   # MongoDB async manager & indexing
│   ├── gateway/                        # Node.js / Express.js API gateway router
│   ├── models/                         # Trained model artifacts & configs
│   │   ├── deberta_fp32_final/         # Fine-tuned DeBERTa-v3 model & tokenizer
│   │   └── hybrid_deberta_tabular/     # Multimodal Fusion Net checkpoints & scalers
│   ├── services/                       # Business logic, DeBERTa, Hybrid & FakeGuard inference
│   └── tests/                          # 29 automated pytest tests (mongomock, health, predict)
├── configs/                            # YAML configuration files for models & training
├── data/
│   ├── splits/                         # Reviewer-held-out train/validation/test partitions
│   └── README.md                       # 17-column dataset documentation & schemas
├── frontend/                           # Client-side web dashboard & scanner UI
├── reports/                            # Detailed evaluation, error analysis & audit reports
│   ├── model_comparison_benchmark.md   # Markdown comparative benchmark report
│   ├── unified_model_benchmark.json    # Serialized unified evaluation metrics
│   └── integration_plan.md             # Integration plan and architecture mapping
├── results/                            # Figures, presentation diagrams, and experiment tables
│   ├── figures/                        # Heatmaps, distributions, and wordclouds
│   └── presentation_diagrams/          # Architecture and problem statement slides
├── scripts/                            # Cross-domain Ott & Salminen benchmarking scripts
├── src/                                # FakeGuard multi-view core package
│   ├── data/                           # Dataset loaders & behavioral feature engineering
│   ├── evaluation/                     # Metrics calculator & explainability visualizers
│   ├── models/                         # FakeGuard, CrossAttentionFusion, GraphEncoder, HybridNet
│   ├── training/                       # Loss functions (Focal Loss, Label Smoothing) & Trainer
│   └── utils/                          # Helper routines
├── training/
│   ├── artifacts/                      # Model checkpoints, scalers, and cache
│   └── scripts/                        # Reproducible training and evaluation pipelines:
│       ├── clean_dataset.py            # 17-column validation & cleaning
│       ├── create_splits.py            # Reviewer-disjoint partitioner
│       ├── train_baseline.py           # TF-IDF + LightGBM
│       ├── train_hybrid_models.py      # DeBERTaTabularFusionNet & LightGBM Hybrid
│       ├── verify_hybrid_inference.py  # Hybrid multimodal verification
│       └── evaluate_unified_models.py  # Unified model evaluation pipeline
├── tests/
│   └── test_model.py                   # Model architecture & loss unit tests
├── requirements.txt                    # Unified root Python dependencies
├── requirements-backend.txt            # Backend service dependencies
├── setup.py                            # Package setup script
├── RESEARCH_AUDIT.md                   # Pramit Ghosh's empirical research audit
├── RESEARCH_PAPER_CHANGES.md           # Detailed research paper changes and methodology
└── README.md                           # Master project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Vinay-rautulwar/fake-review-detection.git
cd fake-review-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unified Model Benchmark
Evaluate all models on the held-out test set:
```bash
python training/scripts/evaluate_unified_models.py
```
This updates `reports/model_comparison_benchmark.md` and `reports/unified_model_benchmark.json`.

### 3. Run Automated Tests
```bash
# Run model architecture and component tests
python -m pytest tests/test_model.py -v

# Run backend API and health tests
python -m pytest backend/tests/test_health.py -v
```

### 4. Start Backend Service
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`
- Web Dashboard: `http://localhost:8000/app`

---

## 📜 Research Documentation

- 📄 [Research Paper Changes & Methodology Document](RESEARCH_PAPER_CHANGES.md)
- 📄 [Empirical Research Audit & Leakage Analysis](RESEARCH_AUDIT.md)
- 📄 [Unified Model Comparison Benchmark](reports/model_comparison_benchmark.md)
- 📄 [Repository Integration Plan](reports/integration_plan.md)

---

## ⚖️ License
This project is open-source under the MIT License.
