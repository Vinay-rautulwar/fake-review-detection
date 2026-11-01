# Integration Plan: Merging Vinay Rautulwar & Pramit Ghosh Repositories

## 1. Executive Summary & Attribution
This integration plan outlines the unification of two distinct, complementary Fake Review Detection projects:
- **User Repository (`Vinay-rautulwar/fake-review-detection`)**:
  - Focus: Production-grade e-commerce review classification (17-column Amazon/e-commerce dataset), DeBERTa-v3-base fine-tuned transformer, reviewer-held-out leakage-free splitting, multimodal tabular + transformer fusion networks (`DeBERTaTabularFusionNet`), threshold calibration, and FastAPI backend service.
- **Friend Repository (`PramitGhosh02/Fake-Review-Detection`)**:
  - Focus: Academic research audit (`RESEARCH_AUDIT.md`), multi-view deep learning framework (`FakeGuard` with text, behavioral, and graph encoders), PyTorch packed-sequence BiLSTM, tuned Linear SVM with TF-IDF n-grams, cross-domain linguistic benchmarking (Ott Deceptive Opinion Corpus, Salminen 40K), SHAP explainability, and Node.js gateway microservice.

**Attribution Policy**:
Both contributors (Vinay Rautulwar and Pramit Ghosh) will be prominently credited across the unified codebase, `README.md`, paper documentation, and git commit history. Neither repository will be overwritten or force-pushed.

---

## 2. Deep Comparative Analysis

| Dimension | Vinay Rautulwar's Project | Pramit Ghosh's Project | Unified Integration Strategy |
| :--- | :--- | :--- | :--- |
| **Primary Dataset** | 30,000-row 17-column e-commerce dataset (29,821 valid rows) with behavioral metadata (reviewer count, rating deviation, verified purchase, etc.). | Ott Deceptive Opinion Corpus (1,600 hotel reviews) + Salminen 40K AI-generated corpus (`CG`/`OR`). | Retain 17-column dataset as primary multimodal benchmark; include Ott corpus and adapters as cross-domain linguistic benchmark. |
| **Data Splitting** | 70/15/15 reviewer-held-out splitting (`train.csv`, `val.csv`, `test.csv`) strictly preventing reviewer identity leakage. | 5-fold cross-validation / hotel-grouped cross-validation (`GroupKFold`) preventing hotel leakage. | Retain reviewer-held-out splits for fair unified model evaluation; incorporate group-aware cross-validation utilities for academic studies. |
| **Models** | - DeBERTa-v3-base (fine-tuned FP32)<br>- TF-IDF + LightGBM<br>- Multimodal DeBERTa + Tabular Fusion Net (Class-Weighted & Focal Loss)<br>- LightGBM Hybrid | - Linear SVM (TF-IDF 1-2 n-grams)<br>- BiLSTM (packed sequences, GloVe/learned embeddings)<br>- FakeGuard Multi-View (Text + MLP Behavior + GAT Graph + View-Attention Fusion)<br>- Logistic Regression & Naive Bayes | Support ALL models in a unified registry: DeBERTa, LightGBM, Linear SVM, BiLSTM, Multimodal Fusion Net, and FakeGuard architecture. |
| **Loss Formulations** | - Cross-Entropy<br>- Class-Weighted Loss ($1.67\times$ penalty)<br>- Focal Loss ($\gamma=2.0, \alpha=0.60$) | - Focal Loss ($\gamma=2.0, \alpha=0.25$)<br>- Label Smoothing Cross-Entropy | Integrate unified losses module supporting Cross-Entropy, Class-Weighted, Focal Loss, and Label Smoothing in PyTorch. |
| **Graph Modeling** | Not in original user repo. | `GraphEncoder` with PyTorch Geometric `GATConv` and fallback mock when PyG is absent. | Port `GraphEncoder` with graceful fallback so graph-based research runs seamlessly without breaking standard environments. |
| **Explainability** | Confusion matrices, error distribution reports, threshold curves. | SHAP wrapper for behavioral features, token attention heatmaps, cross-view attention visualizer. | Integrate SHAP explainability module and attention visualization tools into `src/evaluation/`. |
| **Backend & API** | FastAPI with modular routers, Pydantic v2 schemas, MongoDB repository, health checks, test suite. | FastAPI (adapted from Vinay) + Node.js API Gateway (`server.js`, `review_router.js`). | Preserve robust FastAPI backend as core; include Node.js gateway in `backend/gateway/` for microservices flexibility. |

---

## 3. Integration Architecture & File Mapping

```
fake-review-detection/
├── backend/
│   ├── api/routes/          # FastAPI REST endpoints (predict, health, models, history)
│   ├── gateway/             # [PORTED FROM PRAMIT] Node.js Express API gateway
│   ├── models/              # Checkpoint metadata & configs (DeBERTa, Hybrid, Baseline)
│   └── services/            # Model inference service supporting DeBERTa & Hybrid
├── configs/                 # [PORTED FROM PRAMIT] YAML configs for models and training
├── data/                    # Split scripts, verification, and dataset READMEs
├── docs/                    # Research documentation & integration reports
│   └── RESEARCH_PAPER_INTEGRATION.md # Detailed academic methodology & comparison
├── reports/                 # Comprehensive evaluation & audit reports
├── results/                 # [PORTED FROM PRAMIT] Ott & Salminen experimental results
├── scripts/                 # [PORTED FROM PRAMIT] Ott benchmark & bias audit scripts
├── src/                     # [PORTED FROM PRAMIT] FakeGuard multi-view core package
│   ├── data/                # Dataset loaders & behavioral feature engineering
│   ├── evaluation/          # Metrics calculator, SHAP & attention explainability
│   ├── models/              # FakeGuard, TextEncoder, BehaviorEncoder, GraphEncoder, CrossAttentionFusion
│   ├── training/            # PyTorch Trainer, Focal Loss, Label Smoothing
│   └── utils/               # Helper routines
├── training/
│   ├── artifacts/           # Training checkpoints, cache, scalers
│   └── scripts/             # Data cleaning, split creation, baseline training, hybrid training,
│                            # [NEW] train_svm.py, train_bilstm.py, evaluate_all_models.py
├── tests/                   # Pytest suites for backend and models
└── README.md                # Joint attribution, installation, and unified benchmarks
```

---

## 4. Execution Plan

1. **Safety & Git Configuration**:
   - Verify `.gitignore` ignores binary weights (`*.pt`, `*.safetensors`, `*.npy`) and environment secrets.
   - Ensure local Git history is intact and untracked files are staged properly.

2. **Component Integration**:
   - Port `src/` modular package (`models/`, `data/`, `evaluation/`, `training/`, `utils/`).
   - Port `configs/` (`model_config.yaml`, `train_config.yaml`).
   - Port `scripts/` (`run_ott_experiments.py`, `run_ott_bilstm.py`, `audit_ott_collection_bias.py`, `tune_grouped_svm.py`).
   - Port `results/` (Ott and Salminen result tables and figures).
   - Port `backend/gateway/` (Node.js API gateway).
   - Port `RESEARCH_AUDIT.md`.

3. **Train / Adapt Baseline Models on Unified Dataset**:
   - Adapt and train friend's Linear SVM (`training/scripts/train_svm.py`) on `data/splits/train.csv` / `test.csv`.
   - Adapt and train friend's BiLSTM (`training/scripts/train_bilstm.py`) on `data/splits/train.csv` / `test.csv`.

4. **Unified Model Evaluation Pipeline (`training/scripts/evaluate_all_models.py`)**:
   - Run side-by-side benchmark on the identical test split (4,474 reviews):
     1. TF-IDF + Logistic Regression
     2. TF-IDF + Linear SVM
     3. Word Embeddings + BiLSTM
     4. TF-IDF + LightGBM
     5. DeBERTa-v3 Text-Only ($T = 0.50$)
     6. DeBERTa-v3 Text-Only ($T^* = 0.12$)
     7. LightGBM Multimodal Hybrid
     8. DeBERTa + Tabular Fusion Net (Class-Weighted)
     9. DeBERTa + Tabular Fusion Net (Focal Loss)
   - Save unified benchmark table to `reports/unified_model_benchmark.md` and JSON.

5. **Documentation & Attribution**:
   - Update `README.md` with joint credit to Vinay Rautulwar and Pramit Ghosh, complete project sitemap, and unified performance benchmarks.
   - Update `requirements.txt` and `requirements-backend.txt` to harmonize dependencies.
   - Author `docs/RESEARCH_PAPER_INTEGRATION.md` detailing architectural comparisons, loss functions, and experimental results.

6. **Verification & Push**:
   - Run backend test suite (`pytest backend/tests`).
   - Stage, commit, and push to `origin main` on `https://github.com/Vinay-rautulwar/fake-review-detection.git`.
