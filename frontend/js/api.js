/**
 * Fake Review Detection v2 - REST API Client
 * Interfaces with FastAPI endpoints (/api/v1/*) with strict error handling,
 * truthful model lifecycle detection, and explicitly labelled mock simulation fallback.
 */

const API_CONFIG = {
  // Determine backend base URL: if running on port 8000 use relative path, otherwise default to 127.0.0.1:8000
  baseUrl: (window.location.port === '8000' || window.location.pathname.startsWith('/app'))
    ? '/api/v1'
    : 'http://127.0.0.1:8000/api/v1',
  timeoutMs: 15000,
};

class ApiClient {
  constructor(config = API_CONFIG) {
    this.baseUrl = config.baseUrl;
    this.timeoutMs = config.timeoutMs;
  }

  /**
   * Internal fetch wrapper with timeout and standardized JSON response/error parsing.
   */
  async _fetch(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          ...(options.headers || {}),
        },
      });
      clearTimeout(timeout);

      const contentType = response.headers.get('content-type') || '';
      let data = null;
      if (contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      if (!response.ok) {
        const error = new Error(
          (data && data.detail) ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
          : `HTTP ${response.status}: ${response.statusText}`
        );
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      clearTimeout(timeout);
      if (err.name === 'AbortError') {
        const timeoutErr = new Error(`Request timed out after ${this.timeoutMs / 1000}s. Ensure FastAPI backend is running.`);
        timeoutErr.status = 504;
        throw timeoutErr;
      }
      throw err;
    }
  }

  /**
   * GET /api/v1/health
   * Check system health (API, MongoDB, and configured ML model).
   */
  async getHealth() {
    return this._fetch('/health');
  }

  /**
   * GET /api/v1/models
   * List all configured and supported model architectures.
   */
  async getModels() {
    return this._fetch('/models');
  }

  /**
   * GET /api/v1/models/metrics
   * Retrieve offline validation benchmarks (ROC-AUC, PR-AUC, confusion matrix, threshold comparison).
   */
  async getModelMetrics(modelName = null) {
    const query = modelName ? `/${encodeURIComponent(modelName)}/metrics` : '/metrics';
    return this._fetch(`/models${query}`);
  }

  /**
   * POST /api/v1/predict
   * Classify a single review with optional metadata and historical persistence.
   */
  async predictSingle(payload) {
    return this._fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  /**
   * POST /predict
   * Direct text inference endpoint returning clean { prediction, label, fake_probability, genuine_probability, model }.
   */
  async predictDirect(reviewText) {
    const rootUrl = this.baseUrl.replace(/\/api\/v1\/?$/, '');
    const url = `${rootUrl}/predict`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ review_text: reviewText }),
    });
    const data = await response.json();
    if (!response.ok) {
      const err = new Error((data && data.detail) ? data.detail : `Inference failed with status ${response.status}`);
      err.status = response.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  /**
   * POST /api/v1/batch-predict
   * Upload CSV file for batch classification.
   */
  async predictBatch(file) {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/batch-predict`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    const data = await response.json();
    if (!response.ok) {
      const err = new Error(data.detail || `Batch upload failed with status ${response.status}`);
      err.status = response.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  /**
   * GET /api/v1/history
   * Paginated prediction history from MongoDB.
   */
  async getHistory({ page = 1, pageSize = 15, predictionClass = null, modelName = null, sortOrder = 'desc' } = {}) {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      sort_order: sortOrder,
    });
    if (predictionClass && predictionClass !== 'all') {
      params.append('prediction_class', predictionClass);
    }
    if (modelName) {
      params.append('model_name', modelName);
    }
    return this._fetch(`/history?${params.toString()}`);
  }

  /**
   * GET /api/v1/history/{id}
   * Get single historical prediction detail.
   */
  async getHistoryDetail(predictionId) {
    return this._fetch(`/history/${encodeURIComponent(predictionId)}`);
  }

  /**
   * GET /api/v1/analytics/summary
   * Retrieve MongoDB usage analytics.
   */
  async getAnalyticsSummary() {
    return this._fetch('/analytics/summary');
  }

  /**
   * Explicit Demo/Mock Simulation Mode (Requirement 7)
   * Strictly marked as simulated demo. Never fabricated as real model output.
   */
  simulateDemoPrediction(reviewText, metadata = {}) {
    const textLower = reviewText.toLowerCase();
    
    // Heuristic keywords commonly associated with synthetic/promotional reviews
    const suspiciousKeywords = [
      'exceeded my expectations', '100% recommended', 'must buy', 'best product ever',
      'miracle', 'changed my life', 'flawless', 'five stars', 'unbelievable deal',
      'highly highly recommend', 'dont hesitate', 'buy it now'
    ];
    
    let suspiciousScore = 0.25;
    for (const kw of suspiciousKeywords) {
      if (textLower.includes(kw)) suspiciousScore += 0.18;
    }

    if (metadata.verified_purchase === false) suspiciousScore += 0.15;
    if (metadata.reviewer_review_count && metadata.reviewer_review_count < 2) suspiciousScore += 0.12;

    const fakeProb = Math.min(0.96, Math.max(0.04, suspiciousScore));
    const isFake = fakeProb >= 0.50;
    const confidence = isFake ? fakeProb : (1 - fakeProb);

    return {
      prediction_id: 'demo_' + Math.random().toString(36).substring(2, 10),
      review_text: reviewText,
      predicted_label: isFake ? 'fake' : 'genuine',
      fake_probability: Math.round(fakeProb * 10000) / 10000,
      confidence: Math.round(confidence * 10000) / 10000,
      model_name: 'DEMO_SIMULATION_MODE (Non-ML Heuristic)',
      timestamp: new Date().toISOString(),
      metadata: metadata,
      is_demo: true,
      demo_disclaimer: 'This prediction was generated in DEMO SIMULATION MODE because the backend model is currently training in Google Colab. It is NOT a real ML output.',
    };
  }
}

// Export singleton instance
const api = new ApiClient();
window.api = api;
