# Post-Integration System & Research Audit Report

**Project**: Fake Review Detection System (v2.0.0-integrated)  
**Corpus / Workspace**: `fake-review-detection-v2`  
**Date of Audit**: November 1, 2026  
**Audit Type**: Complete Post-Integration Structural, Architectural, Security, and Research Validity Audit  
**Contributors Unified**:
- **Vinay Rautulwar** (`Vinay-rautulwar/fake-review-detection.git`)
- **Pramit Ghosh** (`PramitGhosh02/Fake-Review-Detection.git`)

---

## Executive Summary & Production-Readiness Verdict

### Production-Readiness Verdict: **NOT PRODUCTION READY**

While core text and tabular multimodal inference modules are operational with verified weights, the project cannot be certified as production-ready due to:
1. **1 Failing Automated Unit Test**: `backend/tests/test_health.py::test_models_list_endpoint` failed (`assert 4 == 3`) due to unupdated test assertions following the registration of `fakeguard`.
2. **Untrained Placeholder Components**: The multi-view `FakeGuard` architecture has **no trained checkpoint file** (`results/checkpoints/best_model.pt` is missing). When requested in the backend, it runs with randomly initialized weights.
3. **Dead Graph Stream**: The Graph Attention Network (GAT) in `FakeGuard` was never trained with graph topology. In `src/models/fakeguard.py` and `scripts/train.py`, graph embeddings default unconditionally to `torch.zeros()`.
4. **Research Validity & Simulation Artifacts**:
   - `scripts/evaluate.py` does not evaluate real data; it uses `np.random.randint(0, 2, 1000)` to generate fake test results.
   - `scripts/generate_figures.py` generates paper figures and LaTeX tables from synthetic numpy distributions rather than trained models.
   - `scripts/train_real_datasets.py` (lines 144–180) synthesizes behavioral metadata conditioned directly on `df['label'] == 1`, introducing catastrophic direct target leakage.
5. **Infrastructure Dependency**: The system requires a running local MongoDB instance for historical persistence and analytics. Single and batch inference degrade gracefully, but historical state tracking is unavailable without MongoDB.

---

## 1. Verified Working Components

The following modules have been thoroughly verified through direct code inspection and runtime execution:

### 1.1 Model Inference Engines & Weights
- **DeBERTa-v3-base (FP32 Text-Only)**:
  - Checkpoint verified on disk: `backend/models/deberta_fp32_final/model.safetensors` (737.7 MB, 86M parameters).
  - Config & Tokenizer verified: `config.json`, `tokenizer.json`, `tokenizer_config.json`.
  - Test benchmark on 4,474 held-out reviews: Accuracy 71.43%, Precision 86.74%, Recall 28.09%, F1 42.43%, ROC-AUC 0.7152, PR-AUC 0.6639.
  - Successfully loads into memory and serves predictions via `backend/services/model_service.py`.
- **TF-IDF + LightGBM Baseline**:
  - Checkpoint verified: `training/artifacts/baseline/lightgbm_model.joblib` (1.3 MB) + `tfidf_vectorizer.joblib` (312 KB).
  - Test benchmark on 4,474 held-out reviews: Accuracy 64.50%, Precision 48.93%, Recall 53.40%, F1 51.07%, ROC-AUC 0.6691, PR-AUC 0.6118.
- **Calibrated Linear SVM (Ott 5-Fold Grouped CV + Unified Test)**:
  - Checkpoint verified: `training/artifacts/svm/calibrated_svm.joblib` (362 KB) + `tfidf_vectorizer.joblib` (664 KB).
  - Metrics verified: Accuracy 64.51%, Precision 63.44%, Recall 12.52%, F1 20.92%, ROC-AUC 0.6210.
- **Packed BiLSTM**:
  - Checkpoint verified: `training/artifacts/bilstm/best_bilstm.pt` (993 KB).
  - Metrics verified: Accuracy 66.76%, Precision 56.51%, Recall 49.19%, F1 52.60%, ROC-AUC 0.7157.
- **DeBERTaTabularFusionNet (Multimodal Deep Fusion)**:
  - Checkpoint verified: `backend/models/hybrid_deberta_tabular/pytorch_hybrid_class_weighted.pt` (457 KB) and `pytorch_hybrid_focal_loss.pt` (457 KB).
  - Feature scaler verified: `backend/models/hybrid_deberta_tabular/tabular_scaler.joblib` (1.2 KB).
  - Test benchmark on 4,474 held-out reviews:
    - Class-Weighted: Accuracy 60.01%, Recall 81.40%, F1 60.41%, ROC-AUC 0.7459, PR-AUC 0.7000.
    - Focal Loss: Accuracy 65.49%, Precision 53.16%, Recall 66.73%, F1 59.18%, ROC-AUC 0.7472, PR-AUC 0.7002.
  - End-to-end verification script `training/scripts/verify_hybrid_inference.py` executed successfully.

### 1.2 Backend API & Database Infrastructure
- **FastAPI Architecture (`backend/main.py`)**:
  - Proper lifespan event management (`lifespan`) for startup/teardown.
  - Standardized JSON responses for all exceptions, completely masking internal tracebacks and database connection details from client responses.
  - Static files properly mounted at `/app` serving `frontend/index.html`.
- **API Endpoints**:
  - `GET /`: Health status and sitemap.
  - `GET /health` & `GET /api/v1/health`: Truthful model readiness and hardware device reporting.
  - `POST /predict` & `POST /api/v1/predict`: Direct prediction supporting both text and metadata.
  - `GET /api/v1/models` & `GET /api/v1/models/metrics`: Returns metadata, architecture details, and verified validation benchmarks.
  - `POST /api/v1/batch-predict`: Handles CSV/JSON file uploads with maximum 500 rows limit and payload validation.
  - `GET /api/v1/history` & `GET /api/v1/analytics/summary`: MongoDB pagination and aggregation pipelines.
- **MongoDB Connection Management (`backend/db/mongodb.py`)**:
  - Graceful degradation: Catches `ServerSelectionTimeoutError` when MongoDB is offline, allowing inference routes to function uninterrupted.
  - Automatic indexing on `timestamp`, `predicted_label`, `model_name`, and compound index `(model_name, predicted_label, timestamp)`.

### 1.3 Dataset Splitting & Leakage Isolation
- **Held-Out Test Set (`data/splits/test.csv`)**:
  - Size: Exactly 4,474 reviews (2,797 Genuine, 1,677 Fake).
  - 100% Reviewer ID isolation across `train.csv`, `validation.csv`, and `test.csv`.
  - Feature scalers fit exclusively on training data.
  - All probability classification thresholds selected exclusively on `validation.csv`.

---

## 2. Incomplete Components & Placeholders

### 2.1 Multi-View FakeGuard Graph Attention Stream
- **Location**: `src/models/fakeguard.py` (lines 98–104), `src/models/encoders.py` (lines 65–118), `scripts/train.py` (lines 219–223).
- **Finding**:
  - In `src/models/fakeguard.py`:
    ```python
    if node_features is not None and edge_index is not None:
        graph_emb = self.graph_encoder(node_features, edge_index)
    else:
        graph_emb = torch.zeros((batch_size, graph_dim), device=input_ids.device)
    ```
  - In `scripts/train.py`: The data loader does not build or load graph tensors (`node_features`, `edge_index`). Batches only contain `input_ids`, `attention_mask`, and `behavioral_features`.
  - **Verdict**: The Graph Attention Network (GAT) encoder is an **untrained placeholder stream**. It was never optimized during training, and always injects zero-vectors into the cross-attention module.

### 2.2 Model Evaluation Script (`scripts/evaluate.py`)
- **Location**: `scripts/evaluate.py` (lines 30–37).
- **Finding**:
  - The script accepts `--checkpoint results/checkpoints/best_model.pt`, prints a status message, and then completely ignores the checkpoint:
    ```python
    print("Running evaluation on test set...")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.clip(y_true * 0.8 + np.random.randn(1000) * 0.3, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    ```
  - **Verdict**: Disconnected placeholder simulation. Does not evaluate any trained weights or real datasets.

### 2.3 Paper Figures & Tables Generator (`scripts/generate_figures.py`)
- **Location**: `scripts/generate_figures.py` (lines 31–64).
- **Finding**:
  - Generates paper figures (ROC curves, ablation bar charts, cross-view attention weights) and LaTeX tables using randomly synthesized numbers:
    ```python
    prob_fg = np.clip(y_true * 0.85 + np.random.randn(1000) * 0.2, 0, 1)
    prob_b1 = np.clip(y_true * 0.70 + np.random.randn(1000) * 0.35, 0, 1)
    ```
  - **Verdict**: Disconnected synthetic asset generator. Does not reflect empirical model evaluation.

---

## 3. Missing Files and Dependencies

### 3.1 Missing Weights & Checkpoint Files
1. `results/checkpoints/best_model.pt`: Missing. (The directory `results/checkpoints/` does not exist).
2. `results/checkpoints/fakeguard_latest.pt`: Missing.
3. `results/checkpoints/cross_attention_fusion.pt`: Missing.

### 3.2 Environment & Dependency Gaps
1. **Python 3.14 Future Warnings**:
   - PyTorch emits `FutureWarning: torch.jit.script is not supported in Python 3.14+ and may break. Please switch to torch.compile or torch.export`.
2. **Starlette Deprecation Warnings**:
   - Deprecation warning on `HTTP_422_UNPROCESSABLE_ENTITY` (should use `HTTP_422_UNPROCESSABLE_CONTENT`).
   - Deprecation warning in test client: `Using httpx with starlette.testclient is deprecated; install httpx2 instead`.

---

## 4. Automated Test Results

### 4.1 Pytest Execution Summary
- **Execution Command**: `python -m pytest backend/tests/ tests/ -v`
- **Total Test Cases**: 41
- **Passed**: 40
- **Failed**: 1
- **Warnings**: 9
- **Execution Time**: 37.99 seconds

### 4.2 Test Failure Details
- **Test File**: `backend/tests/test_health.py`
- **Test Case**: `test_models_list_endpoint`
- **Traceback**:
  ```python
  def test_models_list_endpoint(client: TestClient):
      response = client.get("/api/v1/models")
      assert response.status_code == 200
      data = response.json()
      assert "active_model" in data
      assert "models" in data
  >   assert data["total_models"] == 3
  E   assert 4 == 3
  ```
- **Root Cause**: Following integration, the backend registers 4 models (`deberta_text`, `deberta_fusion`, `baseline`, and `fakeguard`), but the pre-existing test assertion was hardcoded to 3.

### 4.3 Passing Test Suites
- `tests/test_model.py`: 12/12 passed (PyTorch TextEncoder, BehaviorEncoder, GraphEncoder, CrossAttentionFusion, FakeGuard forward passes, FocalLoss, LabelSmoothingLoss).
- `backend/tests/test_predictions.py`: 15/15 passed (Inference, batch predictions, metadata handling, threshold edge cases).
- `backend/tests/test_validation.py`: 9/9 passed (Input bounds, negative prices, out-of-range ratings, history pagination).
- `backend/tests/test_health.py`: 4/5 passed (`test_root_endpoint`, `test_health_endpoint_healthy`, `test_health_endpoint_degraded`, `test_direct_predict_endpoint_empty_and_short_input`).

---

## 5. Security & Sensitive Data Audit

### 5.1 Credential & Secret Management
- **Audit Check**: Inspected `.gitignore`, `.env`, `.env.example`, `backend/core/config.py`, and git history.
- **Result**: **PASS - No credentials leaked.**
  - `.gitignore` strictly rules out `.env`, `.env.*`, `backend/.env`, and `backend/.env.*`.
  - `.env.example` contains only benign local defaults (`mongodb://localhost:27017`).
  - No secret API keys, JWT signing keys, or cloud database credentials exist in the source code or git history.

### 5.2 Model Binaries & Dataset Exclusion in Git
- **Audit Check**: Inspected `.gitignore` for large binaries and raw datasets.
- **Result**: **PASS - Properly Ignored.**
  - Ignored patterns: `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.joblib`, `*.parquet`, `data/raw/*`, `data/splits/*`.
  - Kept lightweight metadata: `config.json`, `tokenizer_config.json`, `validation_metrics.json`, `.gitkeep`.
  - The 737.7 MB `model.safetensors` file is correctly excluded from git tracking.

### 5.3 Exception Sanitization
- Catch-all exception handler in `backend/main.py` catches unhandled server exceptions and returns generic `HTTP 500 Internal Server Error`, preventing database hostnames, stack traces, or environment paths from leaking to external clients.

---

## 6. Research Validity & Data Leakage Audit

### 6.1 Direct Target Leakage in Synthetic Feature Generation
- **Location**: `scripts/train_real_datasets.py` (lines 144–164).
- **Finding**:
  ```python
  fake_mask = df['label'] == 1

  def add_if_missing(col, genuine_dist, fake_dist):
      if col not in df.columns:
          vals = np.where(
              fake_mask,
              fake_dist(n),
              genuine_dist(n)
          )
          df[col] = vals

  add_if_missing('rating_deviation',
      lambda n: np.abs(np.random.normal(0.5, 0.5, n)),   # genuine
      lambda n: np.abs(np.random.normal(1.5, 1.0, n)))   # fake
  ```
- **Severity**: **CRITICAL RESEARCH DEFECT**.
- **Impact**: The behavioral features (`rating_deviation`, `extreme_rating`, `posting_frequency`, `burstiness`) are generated directly from the ground-truth target label `df['label']`. Any machine learning model trained on these features achieves near-perfect classification purely by learning the artificial distribution parameters, completely failing on real-world test data where `label` is unknown.
- **Constraint**: These synthetic features must **never** be cited as real empirical evidence in research papers.

### 6.2 Data Split Consistency & Threshold Selection Protocol
- **Audit Check**: Checked whether all models in `reports/unified_model_benchmark.json` use the identical test split, and whether thresholds were selected using validation data only.
- **Result**: **PASS - Methodologically Rigorous.**
  - All 13 model configurations in `reports/unified_model_benchmark.json` were evaluated against the exact same 4,474 reviews from `data/splits/test.csv`.
  - Decision thresholds (0.12 for DeBERTa, 0.25 for LightGBM, 0.41 for Focal Loss Hybrid, 0.30 for Class-Weighted Hybrid) were calibrated exclusively on `data/splits/validation.csv` by scanning thresholds [0.01, 0.99] to maximize F1. No threshold tuning was performed on `test.csv`.
  - Reviewer isolation is 100% disjoint between train, validation, and test splits.

---

## 7. Comparative Benchmark Summary (Held-Out Test Set: 4,474 Reviews)

| Model Name | Modality | Approach / Origin | Threshold | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | ROC-AUC | PR-AUC | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | Text-Only | Pramit Ghosh Baseline | 0.50 | 63.50% | 51.18% | 57.13% | 53.99% | 0.7089 | 0.6589 | Verified |
| **TF-IDF + Linear SVM** | Text-Only | Pramit Ghosh Tuned SVM | 0.50 | 64.51% | 63.44% | 12.52% | 20.92% | 0.6210 | 0.5167 | Verified |
| **Word Embeddings + BiLSTM** | Text-Only | Pramit Ghosh BiLSTM | 0.50 | 66.76% | 56.51% | 49.19% | 52.60% | 0.7157 | 0.6583 | Verified |
| **TF-IDF + LightGBM Baseline** | Text-Only | Vinay Baseline | 0.50 | 64.50% | 48.93% | 53.40% | 51.07% | 0.6691 | 0.6118 | Verified |
| **TF-IDF + LightGBM Baseline** | Text-Only | Vinay Baseline (Val-Tuned) | 0.25 | 50.29% | 42.59% | **94.33%** | 58.69% | 0.6691 | 0.6118 | Verified |
| **DeBERTa-v3-base** | Text-Only | Vinay Transformer (Default) | 0.50 | **71.43%** | **86.74%** | 28.09% | 42.43% | 0.7152 | 0.6639 | Verified |
| **DeBERTa-v3-base** | Text-Only | Vinay Transformer (Val-Tuned)| 0.12 | 54.29% | 44.68% | 92.19% | 60.19% | 0.7152 | 0.6639 | Verified |
| **LightGBM Multimodal Hybrid** | Multimodal | Vinay Hybrid (Text Prob + Tabular)| 0.50 | 71.37% | 69.11% | 42.70% | 52.78% | 0.7425 | 0.6953 | Verified |
| **DeBERTaTabularFusionNet** | Multimodal | Unified Net (Class-Weighted) | 0.50 | 60.01% | 48.03% | 81.40% | **60.41%** | 0.7459 | 0.7000 | Verified |
| **DeBERTaTabularFusionNet** | Multimodal | Unified Net (Focal Loss) | 0.50 | 65.49% | 53.16% | 66.73% | 59.18% | **0.7472** | **0.7002** | Verified |
| **FakeGuard (Multi-View)** | Multimodal | Pramit Ghosh Architecture | 0.50 | N/A | N/A | N/A | N/A | N/A | N/A | **Untrained** |

---

## 8. Exact Recommended Fixes in Priority Order

### Priority 1 (High - Test & CI Integrity)
- **Fix `backend/tests/test_health.py`**:
  - Update line 63 to assert `data["total_models"] == len(data["models"])` or `data["total_models"] == 4` so automated CI/CD builds pass.

### Priority 2 (High - Research Integrity & Evaluation Truthfulness)
- **Fix `scripts/evaluate.py`**:
  - Replace `np.random.randint` simulation with a true evaluation pipeline that loads `data/splits/test.csv`, instantiates the model weights, and executes inference through PyTorch `torch.no_grad()`.
- **Fix `scripts/generate_figures.py`**:
  - Replace synthetic probability generation with real evaluation metrics from `reports/unified_model_benchmark.json` to generate paper figures and LaTeX tables directly from empirical test runs.

### Priority 3 (High - Data Leakage Remediation)
- **Remove Target-Conditioned Synthesis in `scripts/train_real_datasets.py`**:
  - Eliminate the `fake_mask = df['label'] == 1` dependency.
  - Tabular features must strictly be computed from metadata available at inference time (e.g., character length, word count, exclamation mark ratio, verified purchase boolean).

### Priority 4 (Medium - FakeGuard Graph Pipeline Resolution)
- **Resolve FakeGuard Graph Stream**:
  - Since real e-commerce datasets (Amazon/Ott) do not include full user-product bipartite graphs out of the box, either:
    1. Formally document that FakeGuard operates as a 2-stream (Text + Behavior) multimodal architecture when graph topology is absent.
    2. Build a real graph construction adapter in `src/data/graph_builder.py` connecting reviewers and products via bipartite adjacency matrices.

### Priority 5 (Medium - Train and Checkpoint FakeGuard)
- **Train FakeGuard on Unified Splits**:
  - Execute a clean training run of `FakeGuard` on `data/splits/train.csv` and `data/splits/validation.csv`.
  - Save the resulting weights to `backend/models/fakeguard/best_model.pt` so that backend inference does not rely on random initialization.

### Priority 6 (Low - Production Deployment Hardening)
- **Containerization & Deployment**:
  - Add a multi-container `docker-compose.yml` linking the FastAPI service, MongoDB daemon, and an Nginx reverse proxy.
  - Add an in-memory Redis or LRU cache for high-frequency review classification requests.
