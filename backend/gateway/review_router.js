/**
 * Fake Review Detection v2 - Express Gateway Forwarding Router
 *
 * This router acts as an Express.js integration gateway to forward validated review text
 * to the Python DeBERTa-v3 inference service (FastAPI) running on ML_SERVICE_URL.
 *
 * Configuration:
 *   process.env.ML_SERVICE_URL (default: 'http://127.0.0.1:8000')
 *   process.env.ML_SERVICE_TIMEOUT_MS (default: 5000)
 *
 * Usage:
 *   const express = require('express');
 *   const reviewRouter = require('./gateway/review_router');
 *   const app = express();
 *   app.use(express.json());
 *   app.use('/api', reviewRouter);
 */

const express = require('express');
const router = express.Router();

// Configurable environment settings (no hard-coded secrets or credentials)
const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const TIMEOUT_MS = parseInt(process.env.ML_SERVICE_TIMEOUT_MS, 10) || 5000;

/**
 * Validation Middleware for Review Text Input
 */
function validateReviewInput(req, res, next) {
  const { review_text } = req.body || {};

  if (!review_text || typeof review_text !== 'string') {
    return res.status(400).json({
      error: 'Invalid input: review_text is required and must be a string.',
      status: 400,
    });
  }

  const trimmed = review_text.trim();
  if (trimmed.length < 3) {
    return res.status(400).json({
      error: 'Invalid input: review_text must contain at least 3 non-whitespace characters.',
      status: 400,
    });
  }

  if (trimmed.length > 10000) {
    return res.status(400).json({
      error: 'Invalid input: review_text exceeds maximum limit of 10,000 characters.',
      status: 400,
    });
  }

  req.validatedReviewText = trimmed;
  next();
}

/**
 * POST /predict
 * Forwards validated review text to the Python DeBERTa ML inference service.
 */
router.post('/predict', validateReviewInput, async (req, res) => {
  const targetUrl = `${ML_SERVICE_URL}/predict`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const mlResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ review_text: req.validatedReviewText }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await mlResponse.json();

    if (!mlResponse.ok) {
      return res.status(mlResponse.status).json({
        error: data.detail || 'ML inference service returned an error.',
        status: mlResponse.status,
      });
    }

    // Return the standardized DeBERTa prediction response
    return res.status(200).json({
      prediction: data.prediction,
      label: data.label,
      fake_probability: data.fake_probability,
      genuine_probability: data.genuine_probability,
      model: data.model || 'deberta-v3-base',
    });
  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === 'AbortError') {
      return res.status(504).json({
        error: `ML service request timed out after ${TIMEOUT_MS / 1000} seconds.`,
        status: 504,
      });
    }

    if (err.code === 'ECONNREFUSED' || (err.cause && err.cause.code === 'ECONNREFUSED')) {
      return res.status(503).json({
        error: 'ML inference service is currently offline or unreachable.',
        status: 503,
      });
    }

    return res.status(500).json({
      error: 'Failed to communicate with ML inference service.',
      status: 500,
    });
  }
});

/**
 * GET /health
 * Probes the upstream Python ML service health and returns aggregated status.
 */
router.get('/health', async (req, res) => {
  const targetUrl = `${ML_SERVICE_URL}/health`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  try {
    const mlResponse = await fetch(targetUrl, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const healthData = await mlResponse.json();

    return res.status(mlResponse.status).json({
      gateway_status: 'online',
      ml_service: healthData,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    return res.status(503).json({
      gateway_status: 'online',
      ml_service: {
        status: 'unreachable',
        error: 'Could not connect to Python ML service on ' + ML_SERVICE_URL,
      },
    });
  }
});

module.exports = router;
