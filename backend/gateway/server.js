/**
 * Express Gateway Server Example
 * Run: node server.js
 */
const express = require('express');
const reviewRouter = require('./review_router');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Mount the review forwarding router
app.use('/api', reviewRouter);

app.get('/', (req, res) => {
  res.json({
    service: 'Fake Review Detection Express Gateway',
    version: '2.0.0',
    endpoints: {
      predict: 'POST /api/predict',
      health: 'GET /api/health',
    },
    ml_service_url: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
  });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`[Express Gateway] Listening on http://localhost:${PORT}`);
    console.log(`[Express Gateway] Forwarding to ML service: ${process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000'}`);
  });
}

module.exports = app;
