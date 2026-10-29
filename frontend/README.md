# Frontend Application: Fake Review Detection v2

Production-grade, responsive user interface and AI analytics dashboard for the Fake Review Detection ML system. Built with Vanilla HTML5, Modern CSS (Glassmorphism Dark Theme), and modular JavaScript.

---

## Architecture Overview

```text
frontend/
├── index.html           # Main Single Page Application structure with semantic tabs & ARIA accessibility
├── css/
│   └── styles.css       # Complete dark glassmorphism design system, responsive grid & animations
├── js/
│   ├── api.js           # REST API client (/api/v1/*) with error handling & truthful demo simulation
│   └── app.js           # Application controller, tab routing, live metrics polling, modal viewer
└── README.md            # Frontend architecture & local execution guide
```

---

## Key Dashboard Pages & Features

### 1. Executive Overview Dashboard
- **Component Status**: Live indicator pills for API connectivity, MongoDB persistence, and Active Model state (`Training`, `Model not loaded`, or `Ready`).
- **Metric Cards**: Total Reviews Analyzed, Predicted Fake Reviews, Genuine Reviews, and Active Model runtime specifications.
- **Class Balance**: Live distribution gauge tracking fake vs. genuine review classification proportions.

### 2. Review Analyzer
- **Interactive Review Input**: Textarea with real-time character count enforcement (min 3 characters).
- **Quick-Fill Samples**: Instant-load buttons for Suspicious 5-Star Promotional reviews, Authentic Balanced Critiques, and Short Superlatives.
- **Tabular Metadata Drawer**: Star rating (1.0 to 5.0), Verified Purchase status, Product Category, Price, Reviewer Total Reviews, and Days Since Last Review.
- **Confidence & Probability Gauges**: Visual meter bars color-coded for fast credibility assessment.
- **Truthful Model Handling & Demo Mode**: If the active model is DeBERTa and training in Google Colab, the interface cleanly displays the backend HTTP 503 status with an option to preview with an explicitly watermarked Simulated Demo Mode.

### 3. Batch Review Scanner
- **Drag & Drop CSV Upload**: Supports files containing a `review_text` column (up to 500 rows).
- **Batch Processing**: Displays row-by-row verdicts, fake probabilities, confidence intervals, and error statuses.
- **CSV Export**: Downloadable result table.

### 4. Model Evaluation & Benchmarks
- **Step 5 Baseline Metrics**: Real validation scores (ROC-AUC `0.6691`, PR-AUC `0.6118`, Precision `0.4893`, Recall `0.5340`, F1 `0.5107`).
- **Interactive Threshold Switcher**: Compare Default threshold `0.50` vs Validation-Optimal `0.25`.
- **2x2 Confusion Matrix**: Interactive counts for True Negatives (1,862), False Positives (934), False Negatives (781), True Positives (895).
- **Model Comparison**: TF-IDF + LightGBM vs. DeBERTa-v3-base with truthful "Pending Colab Execution" status markers.
- **Colab Sync Guide**: Instructions for syncing weights into `training/artifacts/deberta_text/`.

### 5. Historical Audit Log
- **Persistent MongoDB Queries**: Real-time queries from `/api/v1/history`.
- **Filtering & Pagination**: Filter by classification class (`all`, `fake`, `genuine`), sort order, and page navigation.
- **Click-to-Inspect Modal**: Detailed view displaying full review text, confidence, timestamp, and metadata.

---

## Running the Application Locally

### Option A: Served Directly via FastAPI Backend (Recommended)
When the FastAPI backend is started, it mounts the `frontend` folder directly at `/app`:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser to:
- **Web Dashboard**: [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app)
- **API Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option B: Standalone Frontend Dev Server (Port 5000)
Run Python's built-in HTTP server from the `frontend/` directory (CORS is pre-configured):

```bash
# Terminal 1: Start FastAPI Backend
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Start Frontend Server
python -m http.server 5000 --directory frontend
```

Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000).
