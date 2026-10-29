# Fake Review Detection ML Application (v2)

An end-to-end Machine Learning system engineered to detect fraudulent and deceptive product reviews by synthesizing contextual textual features with reviewer behavioral metadata.

---

## Repository Structure

```text
fake-review-detection-v2/
├── data/
│   ├── raw/
│   │   └── fake_review_dataset.csv         # Untouched original raw dataset (30,000 rows)
│   ├── processed/
│   │   └── fake_review_dataset_clean.csv   # Cleaned, standardized dataset (29,821 rows)
│   ├── quarantine/
│   │   ├── invalid_rows.csv                # Quarantined invalid rows (179 rows)
│   │   ├── duplicate_reviews.csv           # Quarantined duplicate reviews (0 rows)
│   │   └── near_duplicate_reviews.csv      # Flagged template near-duplicate reviews (3,536 rows)
│   ├── final/
│   │   └── fake_review_dataset_v1.csv      # Quality-gated baseline dataset ready for splitting (29,821 rows)
│   ├── splits/
│   │   ├── train.csv                       # Training split (20,875 rows, 70.0%)
│   │   ├── validation.csv                  # Validation split (4,472 rows, 15.0%)
│   │   └── test.csv                        # Test split (4,474 rows, 15.0%)
│   └── README.md                           # Dataset schema & directory documentation
├── reports/
│   ├── dataset_audit.txt                   # Comprehensive text audit and consistency diagnostic
│   ├── cleaning_log.csv                    # Traceable event log of all excluded rows
│   ├── data_readiness_report.txt           # Gate 3 final dataset quality assessment & verdict
│   ├── split_report.txt                    # Step 4 leakage-aware partition and overlap report
│   ├── baseline_validation_report.txt      # Step 5 TF-IDF + LightGBM baseline validation report
│   └── deberta_validation_report.txt       # Step 6 DeBERTa-v3 validation benchmark & comparison report
├── training/
│   ├── notebooks/
│   │   ├── train_models.ipynb              # Colab-compatible training & validation notebook scaffold
│   │   └── train_deberta.ipynb             # Step 6 DeBERTa-v3-base Colab GPU fine-tuning notebook
│   ├── scripts/
│   │   ├── clean_dataset.py                # Reproducible data cleaning & quarantine pipeline
│   │   ├── validate_dataset.py             # Dataset schema & integrity validator
│   │   ├── create_splits.py                # Leakage-aware train/val/test splitting pipeline
│   │   └── train_baseline.py               # Step 5 TF-IDF + LightGBM baseline training pipeline
│   ├── configs/
│   │   └── training_config.yaml            # Model architecture schemas & training hyperparameters
│   ├── artifacts/
│   │   ├── baseline/                       # Step 5 TF-IDF + LightGBM model, vectorizer, and metrics
│   │   ├── deberta_text/                   # Step 6 DeBERTa-v3 model weights, tokenizer, and metrics
│   │   └── .gitkeep                        # Directory tracking
│   └── requirements-training.txt           # Training environment dependencies
├── frontend/
│   └── README.md                           # Interactive dashboard & review scanner roadmap
├── backend/
│   ├── main.py                         # FastAPI ASGI application, CORS middleware, lifespan events
│   ├── core/
│   │   └── config.py                   # Environment settings loaded via Pydantic Settings
│   ├── api/
│   │   ├── deps.py                     # Dependency injection (services, repos)
│   │   ├── router.py                   # Central API v1 router
│   │   └── routes/                     # Endpoints: health, models, predictions, history, analytics
│   ├── schemas/                        # Pydantic v2 request/response schemas
│   ├── db/                             # PyMongo client lifecycle & index management
│   ├── repositories/                   # MongoDB CRUD and aggregation pipelines
│   ├── services/                       # Business logic, isolated model inference, CSV parsing
│   ├── tests/                          # 25 automated unit and integration tests (mongomock)
│   ├── .env.example                    # Backend environment template
│   └── README.md                       # Backend service documentation
├── database/
│   └── README.md                       # Persistence schema and prediction audit logs
├── .env.example                        # Root environment configuration template
├── requirements-backend.txt            # Backend service dependencies
├── .gitignore                          # Git ignore rules for data, weights, caches, and envs
└── README.md                           # Project setup and documentation
```

---

## Expected Dataset Schema

The validation and training pipelines require a CSV containing exactly **17 columns**:

| # | Column Name | Type | Description |
| :- | :--- | :--- | :--- |
| 1 | `review_id` | str/int | Unique review primary key |
| 2 | `reviewer_id` | str/int | Reviewer identifier |
| 3 | `product_id` | str/int | Product identifier |
| 4 | `review_text` | str | Raw customer review text |
| 5 | `rating` | float | Review star rating (`1.0` to `5.0`) |
| 6 | `verified_purchase` | bool/int | Verified purchase flag (`1`/`0` or `True`/`False`) |
| 7 | `review_date` | str | Review timestamp (e.g. `YYYY-MM-DD`) |
| 8 | `product_category` | str | Product category / department |
| 9 | `product_avg_rating` | float | Product lifetime average rating (`1.0` to `5.0`) |
| 10 | `product_review_count` | int | Total reviews for product (`≥ 0`) |
| 11 | `product_price` | float | Product price (`≥ 0.0`) |
| 12 | `reviewer_review_count` | int | Total reviews by reviewer (`≥ 0`) |
| 13 | `reviewer_avg_rating` | float | Reviewer average rating (`1.0` to `5.0`) |
| 14 | `reviewer_verified_review_count` | int | Verified reviews by reviewer (`≤ reviewer_review_count`) |
| 15 | `reviewer_account_age_days` | int/float | Reviewer account age in days (`≥ 0`) |
| 16 | `days_since_previous_review` | int/float | Days elapsed since reviewer's last review (`≥ 0`) |
| 17 | `label` | int | Binary target label (`0` = Genuine, `1` = Fake) |

---

## How to Run in Google Colab

The training notebook scaffold is specifically designed to be uploaded and run in **Google Colab** with optional GPU acceleration.

### Step 1: Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. In the modal, select the **Upload** tab.
3. Browse and select [`training/notebooks/train_models.ipynb`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/training/notebooks/train_models.ipynb).

### Step 2: (Optional) Select GPU Runtime
Data validation runs smoothly on CPU. For deep learning models (DeBERTa):
1. In Colab, click **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
3. Click **Save**.

### Step 3: Provide Dataset
1. In Colab's left sidebar, click the **Files** (folder) icon.
2. Drag and drop your `reviews_dataset.csv` into the file browser, OR
3. If your dataset is stored in Google Drive, mount your drive in the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATASET_PATH = "/content/drive/MyDrive/path_to_your/reviews_dataset.csv"
   ```

### Step 4: Run the Notebook
Execute cells sequentially:
- **Section 1**: Verifies Python, PyTorch, and GPU availability.
- **Section 2**: Installs minimal training and validation packages.
- **Section 3 & 4**: Configures the dataset path and loads the CSV safely.
- **Section 5**: Runs the full data validation suite.
- **Section 6 & 7**: Loads model configurations and inspects architecture scaffolds.

---

## Local Dataset Validation via CLI

You can validate datasets locally without opening a notebook or starting a server:

```bash
# 1. Install dependencies
pip install -r training/requirements-training.txt

# 2. Run dataset validation
python training/scripts/validate_dataset.py --data_path data/reviews_dataset.csv
```

### Validation Diagnostics Included
- Missing values per column and across rows.
- Duplicate `review_id` detection.
- Exact and normalized duplicate review texts (stripping punctuation, case, and redundant whitespace).
- Ratings out of bounds (`[1.0, 5.0]`) or non-numeric.
- Invalid or non-binary labels (`0` = Genuine, `1` = Fake).
- Class distribution and severe imbalance detection.
- Physical & logical consistency checks (verified count $\le$ total count, non-negative days/prices/counts).

*All validation checks are non-destructive and report findings clearly without deleting records.*

---

## Planned Model Architectures

Defined in [`training/configs/training_config.yaml`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/training/configs/training_config.yaml):

1. **TF-IDF + LightGBM (Step 5 Baseline)**:
   - Fast, interpretable n-gram baseline paired with gradient boosted decision trees.
   - Evaluated on validation split: ROC-AUC = 0.6691, PR-AUC = 0.6118, F1 = 0.5107 (Threshold 0.50).
2. **DeBERTa-v3-base Text-Only (Step 6)**:
   - Deep contextual transformer (`microsoft/deberta-v3-base`) fine-tuned on raw review text.
   - Colab GPU training notebook: [`training/notebooks/train_deberta.ipynb`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/training/notebooks/train_deberta.ipynb).
   - Validation report & baseline comparison: [`reports/deberta_validation_report.txt`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/reports/deberta_validation_report.txt).
3. **DeBERTa-v3-base + Tabular Metadata Fusion (Step 7)**:
   - Multimodal architecture combining transformer text embeddings with reviewer and product behavioral features.

---

## Step 6: DeBERTa-v3-base Text-Only Model Training (Google Colab)

Fine-tuning Microsoft DeBERTa-v3-base (86M parameters) on 20,875 training samples requires GPU acceleration. A dedicated, self-contained notebook is available at [`training/notebooks/train_deberta.ipynb`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/training/notebooks/train_deberta.ipynb).

### Notebook Highlights:
- **Strict Leakage Protection**: Trained only on `data/splits/train.csv` and validated on `data/splits/validation.csv`. The test split (`data/splits/test.csv`) is strictly reserved and never loaded.
- **Text-Only Inputs**: Uses strictly `review_text` mapped to binary target (`genuine=0`, `fake=1`). Reviewer IDs, product IDs, and tabular metadata are omitted.
- **Colab GPU Optimized**: Configured for NVIDIA T4 GPU with mixed precision (FP16), dynamic batch padding, gradient accumulation (effective batch size 16), and early stopping.
- **Drive Persistence**: Checkpoints and model artifacts are saved directly to Google Drive so training can resume if runtime disconnects.
- **Comprehensive Evaluation**: Computes F1, precision, recall, PR-AUC, ROC-AUC, and tunes optimal threshold strictly on validation predictions.

---

## Step 7: FastAPI + MongoDB Backend Service

A modular, production-ready backend built with **FastAPI**, **Pydantic v2**, and **MongoDB** for real-time review inference, batch file classification, and historical tracking.

### 1. Installation & Environment
```bash
# 1. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell (or source venv/bin/activate on Linux/Mac)

# 2. Install backend dependencies
pip install -r requirements-backend.txt

# 3. Configure environment
copy .env.example .env
```

### 2. MongoDB Configuration & Compass
1. **Local MongoDB Server**: Ensure `mongod` is running on `mongodb://localhost:27017` (e.g., via Windows Services or Docker).
2. **MongoDB Compass**: Compass is an interactive GUI client (not the database server). In Compass, click **New Connection** and connect to `mongodb://localhost:27017` to inspect stored documents in the `fake_review_detection` database under collections `predictions` and `model_runs`.

### 3. Start the Backend API & Frontend Dashboard
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Frontend Dashboard: [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app)
- Clean Model Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Clean Direct Inference: `POST http://127.0.0.1:8000/predict`
- Enterprise Health Check: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- Models Status: [http://127.0.0.1:8000/api/v1/models](http://127.0.0.1:8000/api/v1/models)

---

## Step 8: DeBERTa-v3 Model Integration & Direct Inference API

The fine-tuned **microsoft/deberta-v3-base** binary text classifier (trained with FP32 precision on Google Colab) is integrated into the backend under:
`backend/models/deberta_fp32_final/`

### 1. Verified Held-Out Test Metrics
- **ROC-AUC**: 0.7152
- **PR-AUC**: 0.6639
- **Precision (Fake Class)**: 0.8674 (86.74%)
- **Recall (Fake Class)**: 0.2809 (28.09%)
- **F1-Score**: 0.4243 (42.43%)
- **Modality**: Text-only (Processes `review_text` with max sequence length 256; does not directly use tabular metadata).
- *Disclaimer*: Metrics are evaluated strictly on the current held-out dataset and do not guarantee production-grade real-world performance.

### 2. Direct Inference Endpoints

#### `POST /predict`
Request:
```json
{
  "review_text": "The product works exactly as described."
}
```
Response:
```json
{
  "prediction": "genuine",
  "label": 0,
  "fake_probability": 0.2038,
  "genuine_probability": 0.7962,
  "model": "deberta-v3-base"
}
```

#### `GET /health`
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "deberta-v3-base",
  "device": "cpu",
  "precision": "fp32",
  "modality": "text-only"
}
```

---

## Step 9: Express.js Gateway Forwarding Route

If you use an Express.js backend or microservices gateway, a drop-in integration router is provided at [`backend/gateway/review_router.js`](file:///d:/all_app&packages/Antigravity/PBL_5sem/fake-review-detection-v2/backend/gateway/review_router.js):

```javascript
const express = require('express');
const reviewRouter = require('./gateway/review_router');
const app = express();

app.use(express.json());
app.use('/api', reviewRouter);
```
- Forwards validated requests to `process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000'`.
- Includes configurable request timeout (default 5000ms), input validation, and upstream error mapping.

---

## Step 10: Running Automated Tests

Run the full pytest suite (29 unit and integration tests):
```bash
pytest backend/tests/ -v
```

---

## Step 11: Deployment Guidelines (Render & Vercel)

- **Python ML Service (Render Web Service)**:
  - Runtime: Python 3.10+
  - Build Command: `pip install -r requirements-backend.txt`
  - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  - RAM Requirement: Minimum **1 GB - 2 GB RAM** recommended for DeBERTa FP32 model weights.
  - Environment Variables:
    - `MODEL_TYPE=deberta_text`
    - `MODEL_DIR=backend/models/deberta_fp32_final`
    - `MONGODB_URI=<your-mongodb-atlas-uri>`
    - `MONGODB_DATABASE=fake_review_detection`
- **Frontend / Gateway (Vercel)**:
  - Host static frontend (`frontend/`) or Express gateway.
  - Set `ML_SERVICE_URL=https://<your-render-service>.onrender.com`.
  - Note: Vercel serverless functions have a 50MB-250MB size limit and short timeouts, making them unsuitable for hosting large PyTorch models directly. PyTorch inference must run on Render or a dedicated VM/container.


