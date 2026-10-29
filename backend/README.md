# Fake Review Detection API (v2)

Production-ready, modular REST API service for the Fake Review Detection ML system, built with **FastAPI**, **Pydantic v2**, and **MongoDB**.

---

## Architecture Overview

```text
backend/
├── main.py                          # FastAPI ASGI application, CORS middleware, lifespan events
├── core/
│   ├── __init__.py
│   └── config.py                    # Environment settings loaded via Pydantic Settings
├── api/
│   ├── __init__.py
│   ├── deps.py                      # FastAPI dependency injection (services, repos)
│   ├── router.py                    # API v1 central router aggregation
│   └── routes/
│       ├── __init__.py
│       ├── health.py                # GET /api/v1/health (API, DB, model status)
│       ├── models.py                # GET /api/v1/models (Model status and metadata)
│       ├── predictions.py           # POST /api/v1/predict, POST /api/v1/batch-predict
│       ├── history.py               # GET /api/v1/history, GET /api/v1/history/{id}
│       └── analytics.py             # GET /api/v1/analytics/summary
├── schemas/
│   ├── __init__.py
│   ├── common.py                    # Health, Model, Error response schemas
│   └── prediction.py                # Request, Response, Batch, History, Analytics schemas
├── db/
│   ├── __init__.py
│   └── mongodb.py                   # PyMongo client lifecycle, connection pooling, index setup
├── repositories/
│   ├── __init__.py
│   └── prediction_repository.py     # MongoDB CRUD operations & aggregations
├── services/
│   ├── __init__.py
│   ├── model_service.py             # ML model lifecycle, artifact checks, isolated inference
│   └── prediction_service.py        # Business logic, CSV validation, history orchestration
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures (mongomock, test client, model stubs)
    ├── test_health.py               # Health & model metadata tests
    ├── test_predictions.py          # Prediction & batch upload tests
    └── test_validation.py           # Pydantic validation, pagination, and analytics tests
```

---

## 1. Prerequisites & Environment Setup

### Step 1: Create a Python Virtual Environment
```bash
# Navigate to project root
cd fake-review-detection-v2

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Or on Windows (CMD)
.\venv\Scripts\activate.bat
# Or on Linux / macOS
source venv/bin/activate
```

### Step 2: Install Backend Dependencies
```bash
pip install -r requirements-backend.txt
```

---

## 2. MongoDB Setup & Configuration

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Configuration variables in `.env`:
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=fake_review_detection
FLASK_ORIGIN=http://127.0.0.1:5000
MODEL_TYPE=deberta_fusion
MODEL_ARTIFACTS_PATH=
```

### Step 2: Start MongoDB Community Server Locally
Ensure the MongoDB server daemon (`mongod`) is running on `localhost:27017`.
- **Windows (Service)**: Search for "Services" in Windows and confirm `MongoDB Server` is "Running".
- **Windows (Manual)**: Run `mongod --dbpath <path_to_data_folder>`.
- **Docker**: `docker run -d -p 27017:27017 --name mongo-fake-reviews mongo:latest`

> **Note on MongoDB Compass**:
> MongoDB Compass is an interactive graphical viewer, **not** the database server itself. Ensure MongoDB Community Server (`mongod`) is running on port `27017` before opening Compass.
> In Compass, connect using the connection string:
> `mongodb://localhost:27017`
> You will see the database `fake_review_detection` with collections `predictions` and `model_runs` once the backend starts or stores predictions.

---

## 3. Running the FastAPI Server

Start the development server with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Once started:
- **API Root**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## 4. API Endpoints Reference

All API routes are served under prefix `/api/v1`:

### Health & Models
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Returns component-level statuses for API, MongoDB connection, and ML model. |
| `GET` | `/api/v1/models` | Lists all planned model architectures, artifact paths, and active model status. |

### Predictions
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/predict` | Classify a single review (`review_text` required, rating and metadata optional). Returns HTTP 503 if model is not trained. |
| `POST` | `/api/v1/batch-predict` | Upload a CSV containing `review_text` (max 500 rows). Returns predictions and logs to MongoDB. |

### History & Analytics
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/history` | Paginated prediction log with filters (`prediction_class`, `model_name`, `sort_order`). |
| `GET` | `/api/v1/history/{id}` | Retrieve a specific prediction by its 24-character MongoDB `ObjectId`. |
| `GET` | `/api/v1/analytics/summary` | Aggregated prediction metrics (fake %, genuine count, model usage, timeline). |

---

## 5. Model Lifecycle & Truthful Status Reporting

The backend strictly avoids fabricating predictions or heuristic substitutes:
1. **Model Loading**: Models are verified on startup without downloading large weights or training inside FastAPI.
2. **Current Configuration (`MODEL_TYPE=deberta_fusion`)**: Since the DeBERTa Multimodal Fusion model is not trained yet, `/api/v1/models` and `/api/v1/health` report `status: "unavailable"`. Any prediction requests return `HTTP 503 Service Unavailable`.
3. **Baseline Model (`MODEL_TYPE=baseline`)**: If set in `.env`, the system automatically loads the Step 5 trained TF-IDF + LightGBM artifacts from `training/artifacts/baseline/` and serves actual live inferences.

---

## 6. Running Backend Tests

The test suite uses `pytest` and `mongomock` to execute fully isolated tests without requiring real model weights or a running production MongoDB server:

```bash
pytest backend/tests/ -v
```

All 25 automated tests verify:
- Health check states (`healthy`, `degraded` for DB/model disconnects).
- Pydantic schema validation (empty review, out-of-bounds ratings, negative prices).
- Model unavailable HTTP 503 error responses.
- Database unavailable HTTP 503 error responses.
- CSV batch upload limits and missing column validation.
- History pagination, class filtering, and ObjectId format checking.
- Analytics aggregation and distinction from ML benchmark metrics.
