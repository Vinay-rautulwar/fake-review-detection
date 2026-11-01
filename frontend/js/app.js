/**
 * Fake Review Detection v2 - Application Controller
 * Handles UI interactions, routing, live polling, multi-model selection,
 * classification threshold adjustment, cross-model benchmarks, visual analytics,
 * and transparent simulation watermark handling.
 */

class AppController {
  constructor() {
    this.state = {
      activeTab: 'overview',
      systemHealth: null,
      modelsList: [],
      activeModel: 'deberta_text',
      analyticsSummary: null,
      modelMetrics: null,
      benchmarkData: null,
      currentThreshold: 0.50,
      historyPage: 1,
      historyFilter: 'all',
      historyTotalPages: 1,
      batchResults: [],
    };

    // Quick-fill sample reviews
    this.sampleReviews = {
      'suspicious-promo': {
        text: 'This product is absolutely incredible and exceeded every single one of my expectations! Flawless build, arrived in 1 day, changed my life! Highly recommend to everyone, must buy immediately five stars!',
        rating: 5.0,
        verified: 'false',
        category: 'Electronics',
        price: 39.99,
        reviewsCount: 1,
        daysPrev: 0.1,
      },
      'authentic-critique': {
        text: 'I bought this unit three weeks ago. While the battery life lasts around 6 hours as advertised, the plastic hinges feel slightly flimsy. It works well for daily commuting, but keep the receipt in case you need a warranty replacement.',
        rating: 3.5,
        verified: 'true',
        category: 'Electronics',
        price: 89.00,
        reviewsCount: 18,
        daysPrev: 14.2,
      },
      'short-superlative': {
        text: 'Best item ever purchased in my entire life. 100% recommended five stars flawless!',
        rating: 5.0,
        verified: 'false',
        category: 'Home & Kitchen',
        price: 19.99,
        reviewsCount: 2,
        daysPrev: 0.5,
      },
      'metadata-rich': {
        text: 'Received this item after ordering it on a promotional discount. Build quality is solid with good heat dissipation, but the power cord is a bit short. Functions as intended for desktop setups.',
        rating: 4.0,
        verified: 'true',
        category: 'Computers & Accessories',
        price: 54.95,
        reviewsCount: 12,
        daysPrev: 3.8,
      },
    };

    // Descriptions and notes for model selection
    this.modelDescriptions = {
      'deberta_text': 'Processes raw text through fine-tuned DeBERTa-v3-base transformer encoder (86M params, FP32). High precision text classifier.',
      'deberta_fusion': 'Combines DeBERTa-v3 contextual text embeddings with 13 scaled behavioral features via dense fusion net (Focal Loss). Top overall ROC-AUC (0.7472).',
      'baseline': 'LightGBM gradient boosted decision trees on word n-grams (1,2) with 5,000 TF-IDF features. Ultra-fast CPU inference (<5ms).',
      'bilstm': 'Packed sequence bidirectional LSTM with 128d word embeddings and 64d hidden state. Strong sequential linguistic modeling.',
      'svm': 'Linear Support Vector Machine calibrated via CalibratedClassifierCV (Ott 5-Fold). Maximum-margin text decision surface.',
      'fakeguard': 'Tri-view deep learning architecture with cross-attention across Text, Behavioral, and Relational Graph views. Outputs explainable view weights.',
    };

    // Empirical confusion matrices on 4,474 held-out test reviews
    this.confusionMatrixData = {
      'deberta_fusion': {
        model: 'DeBERTa + Tabular Hybrid (FusionNet, 0.50)',
        threshold: '0.50',
        tn: 1811, fp: 986, fn: 558, tp: 1119,
        tn_pct: '64.75%', fp_pct: '35.25%', fn_pct: '33.27%', tp_pct: '66.73%',
        fpr: '35.25%', tpr: '66.73%',
      },
      'deberta_text': {
        model: 'DeBERTa-v3-base Text-Only (FP32, 0.50)',
        threshold: '0.50',
        tn: 2725, fp: 72, fn: 1206, tp: 471,
        tn_pct: '97.43%', fp_pct: '2.57%', fn_pct: '71.91%', tp_pct: '28.09%',
        fpr: '2.57%', tpr: '28.09%',
      },
      'bilstm': {
        model: 'Packed Bidirectional LSTM (BiLSTM, 0.50)',
        threshold: '0.50',
        tn: 2148, fp: 649, fn: 838, tp: 839,
        tn_pct: '76.80%', fp_pct: '23.20%', fn_pct: '49.97%', tp_pct: '50.03%',
        fpr: '23.20%', tpr: '50.03%',
      },
      'baseline': {
        model: 'TF-IDF + LightGBM Baseline (0.50)',
        threshold: '0.50',
        tn: 1990, fp: 807, fn: 781, tp: 896,
        tn_pct: '71.15%', fp_pct: '28.85%', fn_pct: '46.57%', tp_pct: '53.43%',
        fpr: '28.85%', tpr: '53.43%',
      },
      'baseline_tuned': {
        model: 'TF-IDF + LightGBM (Val-Tuned 0.25)',
        threshold: '0.25',
        tn: 665, fp: 2132, fn: 95, tp: 1582,
        tn_pct: '23.78%', fp_pct: '76.22%', fn_pct: '5.66%', tp_pct: '94.34%',
        fpr: '76.22%', tpr: '94.34%',
      },
      'svm': {
        model: 'Calibrated Linear SVM (TF-IDF, 0.50)',
        threshold: '0.50',
        tn: 2465, fp: 332, fn: 1256, tp: 421,
        tn_pct: '88.13%', fp_pct: '11.87%', fn_pct: '74.90%', tp_pct: '25.10%',
        fpr: '11.87%', tpr: '25.10%',
      },
    };

    this.init();
  }

  /**
   * Initialize DOM event listeners and load real-time system state.
   */
  async init() {
    this.bindNavigation();
    this.bindAnalyzerForm();
    this.bindThresholdControls();
    this.bindBatchScanner();
    this.bindEvaluationView();
    this.bindConfusionMatrixSelector();
    this.bindHistoryControls();
    this.bindModal();
    this.bindFigureModal();

    // Initial data fetch
    await this.refreshSystemStatus();
    await this.loadAnalytics();
    await this.loadBenchmarkTable();
    await this.loadRegistryCards();
    await this.loadHistory();

    // Polling interval (every 30 seconds for non-intrusive status updates)
    setInterval(() => this.refreshSystemStatus(), 30000);
  }

  /* ============================================================================
     Navigation & Tab Management
     ============================================================================ */
  bindNavigation() {
    const navButtons = document.querySelectorAll('.nav-tab-btn');
    navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        this.switchTab(tabName);
      });
    });
  }

  switchTab(tabName) {
    this.state.activeTab = tabName;

    // Update nav buttons
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
      const isTarget = btn.getAttribute('data-tab') === tabName;
      btn.classList.toggle('active', isTarget);
      btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
    });

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
      pane.classList.toggle('active', pane.id === `pane-${tabName}`);
    });

    // Tab-specific refreshes
    if (tabName === 'history') {
      this.loadHistory();
    } else if (tabName === 'overview') {
      this.loadAnalytics();
    } else if (tabName === 'evaluation') {
      this.loadBenchmarkTable();
    } else if (tabName === 'registry') {
      this.loadRegistryCards();
    }
  }

  /* ============================================================================
     System Status & Model Lifecycle
     ============================================================================ */
  async refreshSystemStatus() {
    try {
      const health = await window.api.getHealth();
      this.state.systemHealth = health;

      const modelsData = await window.api.getModels();
      this.state.modelsList = modelsData.models || [];
      this.state.activeModel = modelsData.active_model;

      this.updateHeaderBadges(health, modelsData);
      this.updateOverviewModelSpecs(modelsData);
    } catch (err) {
      console.warn('Backend connectivity warning:', err);
      this.renderBackendOfflineState();
    }
  }

  updateHeaderBadges(health, modelsData) {
    const dbPill = document.getElementById('db-status-pill');
    const dbText = document.getElementById('db-status-text');
    const modelPill = document.getElementById('model-status-pill');
    const modelText = document.getElementById('model-status-text');

    // 1. Database Badge
    if (health.database && health.database.status === 'connected') {
      dbPill.className = 'status-pill ready';
      dbText.textContent = 'DB: Connected';
    } else {
      dbPill.className = 'status-pill unavailable';
      dbText.textContent = 'DB: Offline';
    }

    // 2. Model Badge
    const activeModelMeta = this.state.modelsList.find(m => m.name === modelsData.active_model);
    const isAvailable = activeModelMeta ? activeModelMeta.is_available : true;
    const isLoaded = activeModelMeta ? activeModelMeta.is_loaded : true;
    const modelName = modelsData.active_model || 'deberta_text';

    if (isAvailable && isLoaded) {
      modelPill.className = 'status-pill ready';
      modelText.textContent = `Model: Ready (${modelName})`;
    } else {
      modelPill.className = 'status-pill ready';
      modelText.textContent = `Model: ${modelName}`;
    }
  }

  updateOverviewModelSpecs(modelsData) {
    const activeKey = modelsData.active_model;
    const modelMeta = this.state.modelsList.find(m => m.name === activeKey);

    const nameEl = document.getElementById('spec-model-name');
    const cardActiveName = document.getElementById('card-active-model-name');

    if (nameEl) nameEl.textContent = activeKey;
    if (cardActiveName && modelMeta) {
      cardActiveName.textContent = modelMeta.display_name || modelMeta.name;
    }
  }

  renderBackendOfflineState() {
    const dbPill = document.getElementById('db-status-pill');
    const dbText = document.getElementById('db-status-text');
    const modelPill = document.getElementById('model-status-pill');
    const modelText = document.getElementById('model-status-text');

    if (dbPill) {
      dbPill.className = 'status-pill unavailable';
      dbText.textContent = 'API / DB: Unreachable';
    }
    if (modelPill) {
      modelPill.className = 'status-pill unavailable';
      modelText.textContent = 'Model: Offline';
    }
  }

  /* ============================================================================
     Overview Analytics
     ============================================================================ */
  async loadAnalytics() {
    try {
      const summary = await window.api.getAnalyticsSummary();
      this.state.analyticsSummary = summary;

      const totalEl = document.getElementById('card-total-reviews');
      const fakeEl = document.getElementById('card-fake-reviews');
      const fakeSubEl = document.getElementById('card-fake-pct-subtitle');
      const genuineEl = document.getElementById('card-genuine-reviews');
      const genuineSubEl = document.getElementById('card-genuine-pct-subtitle');

      const barFake = document.getElementById('bar-fake');
      const barGenuine = document.getElementById('bar-genuine');
      const legendFake = document.getElementById('legend-fake-val');
      const legendGenuine = document.getElementById('legend-genuine-val');

      if (totalEl) totalEl.textContent = summary.total_predictions.toLocaleString();
      if (fakeEl) fakeEl.textContent = summary.fake_count.toLocaleString();
      if (fakeSubEl) fakeSubEl.textContent = `${summary.fake_percentage}% of total submissions`;

      const genuineCount = summary.genuine_count;
      const genuinePct = summary.total_predictions > 0
        ? Math.round((genuineCount / summary.total_predictions) * 1000) / 10
        : 0.0;

      if (genuineEl) genuineEl.textContent = genuineCount.toLocaleString();
      if (genuineSubEl) genuineSubEl.textContent = `${genuinePct}% of total submissions`;

      if (legendFake) legendFake.textContent = `${summary.fake_count} (${summary.fake_percentage}%)`;
      if (legendGenuine) legendGenuine.textContent = `${genuineCount} (${genuinePct}%)`;

      if (barFake && barGenuine) {
        if (summary.total_predictions === 0) {
          barFake.style.width = '50%';
          barGenuine.style.width = '50%';
        } else {
          barFake.style.width = `${summary.fake_percentage}%`;
          barGenuine.style.width = `${100 - summary.fake_percentage}%`;
        }
      }
    } catch (err) {
      console.log('Notice: MongoDB predictions query returned:', err.message);
    }
  }

  /* ============================================================================
     Review Analyzer Controls & Form
     ============================================================================ */
  bindAnalyzerForm() {
    const form = document.getElementById('review-analyzer-form');
    const textarea = document.getElementById('review-text-input');
    const charCounter = document.getElementById('review-char-count');
    const clearBtn = document.getElementById('clear-input-btn');
    const accordionToggle = document.getElementById('meta-accordion-toggle');
    const accordionContent = document.getElementById('meta-accordion-content');
    const sampleButtons = document.querySelectorAll('.sample-btn');
    const modelSelect = document.getElementById('analyzer-model-select');

    // Model selection change handler
    if (modelSelect) {
      modelSelect.addEventListener('change', () => {
        const selectedKey = modelSelect.value;
        const tagEl = document.getElementById('selected-model-status-tag');
        const descEl = document.getElementById('analyzer-model-desc');

        if (descEl) descEl.textContent = this.modelDescriptions[selectedKey] || '';
        if (tagEl) {
          if (selectedKey === 'fakeguard') {
            tagEl.className = 'table-badge experimental';
            tagEl.textContent = 'Experimental';
          } else {
            tagEl.className = 'table-badge genuine';
            tagEl.textContent = 'Operational';
          }
        }
      });
    }

    // Live character counter
    if (textarea && charCounter) {
      textarea.addEventListener('input', () => {
        const count = textarea.value.trim().length;
        charCounter.textContent = `${count} char${count === 1 ? '' : 's'}`;
        charCounter.classList.toggle('valid', count >= 3);
      });
    }

    // Metadata accordion toggle
    if (accordionToggle && accordionContent) {
      accordionToggle.addEventListener('click', () => {
        const isOpen = accordionContent.classList.toggle('open');
        accordionToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      });
    }

    // Quick-fill sample buttons
    sampleButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const sampleKey = btn.getAttribute('data-sample');
        const sample = this.sampleReviews[sampleKey];
        if (sample && textarea) {
          textarea.value = sample.text;
          textarea.dispatchEvent(new Event('input'));

          const ratingInput = document.getElementById('meta-rating');
          const verifiedInput = document.getElementById('meta-verified');
          const categoryInput = document.getElementById('meta-category');
          const priceInput = document.getElementById('meta-price');
          const countInput = document.getElementById('meta-reviewer-count');
          const daysInput = document.getElementById('meta-days-prev');

          if (ratingInput) ratingInput.value = sample.rating || '';
          if (verifiedInput) verifiedInput.value = sample.verified || '';
          if (categoryInput) categoryInput.value = sample.category || '';
          if (priceInput) priceInput.value = sample.price || '';
          if (countInput) countInput.value = sample.reviewsCount || '';
          if (daysInput) daysInput.value = sample.daysPrev || '';

          if (accordionContent && !accordionContent.classList.contains('open')) {
            accordionContent.classList.add('open');
            accordionToggle.setAttribute('aria-expanded', 'true');
          }
        }
      });
    });

    // Clear button
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        if (textarea) {
          textarea.value = '';
          textarea.dispatchEvent(new Event('input'));
        }
        document.getElementById('meta-rating').value = '';
        document.getElementById('meta-verified').value = '';
        document.getElementById('meta-category').value = '';
        document.getElementById('meta-price').value = '';
        document.getElementById('meta-reviewer-count').value = '';
        document.getElementById('meta-days-prev').value = '';

        this.setAnalyzerState('empty');
      });
    }

    // Form submit handler
    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.handleAnalysisSubmit();
      });
    }
  }

  bindThresholdControls() {
    const slider = document.getElementById('analyzer-threshold-slider');
    const displayVal = document.getElementById('threshold-display-val');
    const presetPills = document.querySelectorAll('.preset-pill');

    if (slider && displayVal) {
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        this.state.currentThreshold = val;
        displayVal.textContent = val.toFixed(2);

        // Highlight preset if matching
        presetPills.forEach(pill => {
          const pillVal = parseFloat(pill.getAttribute('data-thresh'));
          pill.classList.toggle('active', Math.abs(pillVal - val) < 0.005);
        });
      });
    }

    presetPills.forEach(pill => {
      pill.addEventListener('click', () => {
        const val = parseFloat(pill.getAttribute('data-thresh'));
        if (slider) {
          slider.value = val;
          slider.dispatchEvent(new Event('input'));
        }
      });
    });
  }

  async handleAnalysisSubmit() {
    if (this.isSubmitting) return;

    const textarea = document.getElementById('review-text-input');
    const text = textarea ? textarea.value.trim() : '';

    if (!text || text.length < 3) {
      this.renderAnalyzerError({
        status: 400,
        message: 'Review content must contain at least 3 non-whitespace characters.',
      });
      return;
    }

    // Read selected model and threshold
    const modelSelect = document.getElementById('analyzer-model-select');
    const selectedModel = modelSelect ? modelSelect.value : 'deberta_text';
    const slider = document.getElementById('analyzer-threshold-slider');
    const threshold = slider ? parseFloat(slider.value) : 0.50;

    // Collect optional metadata
    const ratingVal = document.getElementById('meta-rating').value;
    const verifiedVal = document.getElementById('meta-verified').value;
    const categoryVal = document.getElementById('meta-category').value.trim();
    const priceVal = document.getElementById('meta-price').value;
    const reviewerCountVal = document.getElementById('meta-reviewer-count').value;
    const daysPrevVal = document.getElementById('meta-days-prev').value;

    const payload = {
      review_text: text,
      model: selectedModel,
      threshold: threshold,
    };
    if (ratingVal) payload.rating = parseFloat(ratingVal);
    if (verifiedVal !== '') payload.verified_purchase = verifiedVal === 'true';
    if (categoryVal) payload.product_category = categoryVal;
    if (priceVal) payload.product_price = parseFloat(priceVal);
    if (reviewerCountVal) payload.reviewer_review_count = parseInt(reviewerCountVal, 10);
    if (daysPrevVal) payload.days_since_previous_review = parseFloat(daysPrevVal);

    this.isSubmitting = true;
    this.setAnalyzerState('loading');

    try {
      const result = await window.api.predictSingle(payload);
      this.renderPredictionResult(result, selectedModel, threshold);
      this.loadAnalytics();
      this.loadHistory();
    } catch (err) {
      console.warn('Backend inference call rejected:', err);
      this.renderAnalyzerError(err);
    } finally {
      this.isSubmitting = false;
      const submitBtn = document.getElementById('submit-analysis-btn');
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  setAnalyzerState(state) {
    const emptyEl = document.getElementById('result-empty-state');
    const loadingEl = document.getElementById('result-loading-state');
    const contentEl = document.getElementById('result-content');
    const errorEl = document.getElementById('result-error-state');
    const submitBtn = document.getElementById('submit-analysis-btn');

    if (emptyEl) emptyEl.style.display = (state === 'empty') ? 'flex' : 'none';
    if (loadingEl) loadingEl.style.display = (state === 'loading') ? 'flex' : 'none';
    if (contentEl) contentEl.style.display = (state === 'result') ? 'block' : 'none';
    if (errorEl) errorEl.style.display = (state === 'error') ? 'flex' : 'none';

    if (submitBtn) submitBtn.disabled = (state === 'loading');
  }

  renderPredictionResult(result, selectedModel, threshold) {
    this.setAnalyzerState('result');

    const watermark = document.getElementById('experimental-watermark-pill');
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const verdictIconSvg = document.getElementById('verdict-icon-svg');

    const confVal = document.getElementById('gauge-confidence-val');
    const confMeter = document.getElementById('meter-confidence-fill');
    const fakeVal = document.getElementById('gauge-fake-val');
    const fakeMeter = document.getElementById('meter-fake-fill');
    const genuineLabel = document.getElementById('gauge-genuine-label');
    const threshInfo = document.getElementById('gauge-threshold-info');

    const modelNameEl = document.getElementById('result-model-name');
    const threshUsedEl = document.getElementById('result-threshold-used');
    const timestampEl = document.getElementById('result-timestamp');
    const badgeEl = document.getElementById('result-model-badge');

    // Experimental label for FakeGuard
    const isFakeGuard = selectedModel === 'fakeguard' || result.model_name === 'fakeguard';
    if (watermark) {
      watermark.style.display = isFakeGuard ? 'flex' : 'none';
    }

    const isFake = (result.predicted_label === 'fake' || result.prediction === 'fake' || result.label === 1);
    const fakeProb = (result.fake_probability !== undefined) ? result.fake_probability : (isFake ? result.confidence : 1.0 - result.confidence);
    const genuineProb = (result.genuine_probability !== undefined) ? result.genuine_probability : (1.0 - fakeProb);
    const confidence = (result.confidence !== undefined) ? result.confidence : Math.max(fakeProb, genuineProb);

    const confPct = Math.round(confidence * 1000) / 10;
    const fakePct = Math.round(fakeProb * 1000) / 10;
    const genuinePct = Math.round(genuineProb * 1000) / 10;

    // Verdict Banner
    if (verdictBanner) {
      verdictBanner.className = `verdict-banner ${isFake ? 'fake' : 'genuine'}`;
    }
    if (verdictTitle) {
      verdictTitle.textContent = isFake ? 'FLAGGED DECEPTIVE / FAKE' : 'CLASSIFIED GENUINE / AUTHENTIC';
    }
    if (verdictDesc) {
      verdictDesc.textContent = isFake
        ? `Model probability P(Fake) = ${(fakeProb * 100).toFixed(1)}% exceeds decision threshold τ = ${threshold.toFixed(2)}.`
        : `Model probability P(Fake) = ${(fakeProb * 100).toFixed(1)}% remains below decision threshold τ = ${threshold.toFixed(2)}.`;
    }
    if (verdictIconSvg) {
      verdictIconSvg.innerHTML = isFake
        ? '<path d="M18 6L6 18M6 6l12 12"></path>'
        : '<polyline points="20 6 9 17 4 12"></polyline>';
    }

    // Gauges
    if (confVal) confVal.textContent = `${confPct}%`;
    if (confMeter) confMeter.style.width = `${confPct}%`;
    if (genuineLabel) genuineLabel.textContent = `Genuine Prob: ${genuinePct}%`;
    if (threshInfo) threshInfo.textContent = `Applied Threshold: τ = ${threshold.toFixed(2)}`;

    if (fakeVal) {
      fakeVal.textContent = `${fakePct}%`;
      fakeVal.style.color = isFake ? 'var(--rose-400)' : 'var(--emerald-400)';
    }
    if (fakeMeter) {
      fakeMeter.className = `meter-fill ${isFake ? 'rose' : 'emerald'}`;
      fakeMeter.style.width = `${fakePct}%`;
    }

    // Result metadata tags
    const activeName = result.model || result.model_name || selectedModel;
    if (modelNameEl) modelNameEl.textContent = activeName;
    if (threshUsedEl) threshUsedEl.textContent = `τ = ${threshold.toFixed(2)}`;
    if (timestampEl) timestampEl.textContent = result.timestamp ? new Date(result.timestamp).toUTCString() : new Date().toUTCString();
    if (badgeEl) {
      badgeEl.textContent = isFakeGuard ? 'EXPERIMENTAL CHECKPOINT' : 'LIVE INFERENCE READY';
      badgeEl.style.color = isFakeGuard ? 'var(--amber-400)' : 'var(--emerald-400)';
    }

    // FakeGuard View-Attention Matrix Rendering
    const attentionCard = document.getElementById('fakeguard-attention-card');
    const attentionGrid = document.getElementById('attention-matrix-grid');

    if (attentionCard && attentionGrid) {
      if (isFakeGuard && result.attention_weights) {
        attentionCard.style.display = 'block';
        attentionGrid.innerHTML = '';

        const views = ['Text', 'Behavior', 'Graph'];
        const matrix = result.attention_weights;

        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            const rawVal = (matrix[r] && matrix[r][c] !== undefined) ? matrix[r][c] : 0.333;
            const pct = Math.round(rawVal * 100);
            const cell = document.createElement('div');
            cell.className = 'attention-cell';
            cell.innerHTML = `
              <div class="attention-cell-val">${pct}%</div>
              <div class="attention-cell-lbl">${views[r]} &rarr; ${views[c]}</div>
            `;
            attentionGrid.appendChild(cell);
          }
        }
      } else {
        attentionCard.style.display = 'none';
      }
    }
  }

  renderAnalyzerError(err) {
    this.setAnalyzerState('error');

    const titleEl = document.getElementById('error-state-title');
    const msgEl = document.getElementById('error-state-message');
    const badgeEl = document.getElementById('result-model-badge');

    if (titleEl) {
      titleEl.textContent = (err.status === 503) ? 'Model Unavailable (HTTP 503)' : 'Inference Request Error';
    }
    if (msgEl) {
      msgEl.textContent = err.message || 'The backend cannot serve predictions with the current model configuration.';
    }
    if (badgeEl) {
      badgeEl.textContent = `HTTP ${err.status || 500}`;
      badgeEl.style.color = 'var(--rose-400)';
    }
  }

  /* ============================================================================
     Batch Review Scanner
     ============================================================================ */
  bindBatchScanner() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('batch-file-input');
    const downloadBtn = document.getElementById('download-batch-csv-btn');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          this.handleBatchFile(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
          this.handleBatchFile(fileInput.files[0]);
        }
      });
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.exportBatchResultsCsv());
    }
  }

  async handleBatchFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Invalid file format. Please upload a .csv file.');
      return;
    }

    const progressBox = document.getElementById('batch-progress-box');
    const fileNameEl = document.getElementById('batch-file-name');
    const statusEl = document.getElementById('batch-progress-status');

    if (progressBox) progressBox.style.display = 'block';
    if (fileNameEl) fileNameEl.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    if (statusEl) {
      statusEl.className = 'status-pill training';
      statusEl.textContent = 'Processing Batch Inference...';
    }

    const modelSelect = document.getElementById('batch-model-select');
    const selectedModel = modelSelect ? modelSelect.value : 'deberta_text';
    const threshInput = document.getElementById('batch-threshold-input');
    const threshold = threshInput ? parseFloat(threshInput.value) : 0.50;

    try {
      const data = await window.api.predictBatch(file, selectedModel, threshold);
      this.state.batchResults = data.predictions || [];

      if (statusEl) {
        statusEl.className = 'status-pill ready';
        statusEl.textContent = 'Batch Inference Complete';
      }

      this.renderBatchResults(data);
      this.loadAnalytics();
      this.loadHistory();
    } catch (err) {
      console.warn('Batch processing failed:', err);
      if (statusEl) {
        statusEl.className = 'status-pill unavailable';
        statusEl.textContent = 'Batch Failed';
      }
      alert(`Batch processing error: ${err.message}`);
    }
  }

  renderBatchResults(data) {
    const panel = document.getElementById('batch-results-panel');
    const summaryBadge = document.getElementById('batch-summary-badge');
    const tbody = document.getElementById('batch-results-tbody');

    if (!panel || !tbody) return;

    panel.style.display = 'block';
    const fakeCount = (data.predictions || []).filter(p => p.predicted_label === 'fake').length;
    const genuineCount = (data.predictions || []).filter(p => p.predicted_label === 'genuine').length;

    if (summaryBadge) {
      summaryBadge.textContent = `${data.total_successful} Processed (${fakeCount} Deceptive, ${genuineCount} Genuine) of ${data.total_submitted}`;
    }

    tbody.innerHTML = '';
    (data.predictions || []).slice(0, 100).forEach((item, idx) => {
      const tr = document.createElement('tr');
      const isFake = item.predicted_label === 'fake';
      const isSuccess = item.status === 'success';

      tr.innerHTML = `
        <td class="font-mono">${item.row_index + 1}</td>
        <td style="max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${this.escapeHtml(item.review_text)}
        </td>
        <td>
          ${isSuccess 
            ? `<span class="table-badge ${isFake ? 'fake' : 'genuine'}">${isFake ? 'Fake' : 'Genuine'}</span>`
            : '<span class="table-badge unavailable">Error</span>'
          }
        </td>
        <td class="font-mono">${item.fake_probability !== null ? (item.fake_probability * 100).toFixed(1) + '%' : '—'}</td>
        <td class="font-mono">${item.confidence !== null ? (item.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td>
          <span style="font-size: 0.75rem; color: ${isSuccess ? 'var(--emerald-400)' : 'var(--rose-400)'};">
            ${isSuccess ? 'Success' : 'Failed'}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  exportBatchResultsCsv() {
    if (!this.state.batchResults || this.state.batchResults.length === 0) {
      alert('No batch results available to export.');
      return;
    }

    let csvContent = 'row_index,review_text,predicted_label,fake_probability,confidence,status,error\n';
    this.state.batchResults.forEach(item => {
      const cleanText = (item.review_text || '').replace(/"/g, '""');
      csvContent += `${item.row_index},"${cleanText}",${item.predicted_label || ''},${item.fake_probability ?? ''},${item.confidence ?? ''},${item.status},"${item.error || ''}"\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `fake_review_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /* ============================================================================
     Model Evaluation & Benchmarks
     ============================================================================ */
  bindEvaluationView() {
    const evalButtons = document.querySelectorAll('.eval-tab-btn');
    evalButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const evalKey = btn.getAttribute('data-eval');
        evalButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.eval-content-view').forEach(view => {
          view.style.display = (view.id === `eval-view-${evalKey}`) ? 'block' : 'none';
        });
      });
    });
  }

  bindConfusionMatrixSelector() {
    const select = document.getElementById('cm-model-select');
    if (!select) return;

    select.addEventListener('change', () => {
      const key = select.value;
      const data = this.confusionMatrixData[key];
      if (!data) return;

      const tnEl = document.getElementById('interactive-cm-tn');
      const fpEl = document.getElementById('interactive-cm-fp');
      const fnEl = document.getElementById('interactive-cm-fn');
      const tpEl = document.getElementById('interactive-cm-tp');

      const tnSub = document.getElementById('interactive-cm-tn-pct');
      const fpSub = document.getElementById('interactive-cm-fp-pct');
      const fnSub = document.getElementById('interactive-cm-fn-pct');
      const tpSub = document.getElementById('interactive-cm-tp-pct');

      const diagModel = document.getElementById('cm-diag-model');
      const diagThresh = document.getElementById('cm-diag-thresh');
      const diagFpr = document.getElementById('cm-diag-fpr');
      const diagTpr = document.getElementById('cm-diag-tpr');

      if (tnEl) tnEl.textContent = data.tn.toLocaleString();
      if (fpEl) fpEl.textContent = data.fp.toLocaleString();
      if (fnEl) fnEl.textContent = data.fn.toLocaleString();
      if (tpEl) tpEl.textContent = data.tp.toLocaleString();

      if (tnSub) tnSub.textContent = data.tn_pct;
      if (fpSub) fpSub.textContent = data.fp_pct;
      if (fnSub) fnSub.textContent = data.fn_pct;
      if (tpSub) tpSub.textContent = data.tp_pct;

      if (diagModel) diagModel.textContent = data.model;
      if (diagThresh) diagThresh.textContent = `τ = ${data.threshold}`;
      if (diagFpr) diagFpr.textContent = data.fpr;
      if (diagTpr) diagTpr.textContent = data.tpr;
    });
  }

  async loadBenchmarkTable() {
    const tbody = document.getElementById('unified-benchmark-tbody');
    if (!tbody) return;

    try {
      const data = await window.api.getBenchmarkData();
      this.state.benchmarkData = data;

      tbody.innerHTML = '';
      const entries = Object.entries(data);

      if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-muted);">No benchmark metrics loaded.</td></tr>';
        return;
      }

      entries.forEach(([modelName, details]) => {
        const m = details.metrics || {};
        const tr = document.createElement('tr');

        const acc = (m.accuracy !== undefined) ? (m.accuracy * 100).toFixed(2) + '%' : '—';
        const prec = (m.precision !== undefined) ? (m.precision * 100).toFixed(2) + '%' : '—';
        const rec = (m.recall !== undefined) ? (m.recall * 100).toFixed(2) + '%' : '—';
        const f1 = (m.f1 !== undefined) ? (m.f1 * 100).toFixed(2) + '%' : '—';
        const rocAuc = (m.roc_auc !== undefined) ? m.roc_auc.toFixed(4) : '—';
        const prAuc = (m.pr_auc !== undefined) ? m.pr_auc.toFixed(4) : '—';
        const thresh = (details.threshold !== undefined) ? `τ = ${details.threshold.toFixed(2)}` : '0.50';

        // Highlight top performers
        const isTopAuc = m.roc_auc && m.roc_auc >= 0.74;
        const isTopPrec = m.precision && m.precision >= 0.85;

        tr.innerHTML = `
          <td><strong>${this.escapeHtml(modelName)}</strong></td>
          <td style="font-size: 0.8rem; color: var(--text-muted);">${this.escapeHtml(details.modality || details.approach || 'Unified')}</td>
          <td class="font-mono">${thresh}</td>
          <td class="font-mono">${acc}</td>
          <td class="font-mono ${isTopPrec ? 'benchmark-highlight' : ''}">${prec}</td>
          <td class="font-mono">${rec}</td>
          <td class="font-mono">${f1}</td>
          <td class="font-mono ${isTopAuc ? 'benchmark-highlight' : ''}">${rocAuc}</td>
          <td class="font-mono ${isTopAuc ? 'benchmark-highlight' : ''}">${prAuc}</td>
          <td><span class="table-badge genuine" style="font-size: 0.7rem;">Held-Out Test</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.warn('Benchmark table loading error:', err);
    }
  }

  /* ============================================================================
     Model Registry Catalog
     ============================================================================ */
  async loadRegistryCards() {
    const container = document.getElementById('model-registry-cards-container');
    if (!container) return;

    try {
      const data = await window.api.getModels();
      const models = data.models || [];

      // 12 comprehensive models across both repositories
      const catalog = [
        {
          name: 'DeBERTa-v3-base Text-Only (FP32)',
          id: 'deberta_text',
          category: 'Transformer Representation Learning',
          contributor: 'Vinay Rautulwar',
          status: 'operational',
          modality: 'Raw Review Body (max 256 tokens)',
          architecture: 'microsoft/deberta-v3-base (86M params, FP32)',
          checkpoint: 'backend/models/deberta_fp32_final/',
          dataset: 'Clean Unified Dataset (Held-out Test)',
          metrics: 'ROC-AUC: 0.7152 • Precision: 86.74% • Accuracy: 71.43%',
          active: true,
        },
        {
          name: 'DeBERTa + Tabular Hybrid (FusionNet)',
          id: 'deberta_fusion',
          category: 'Multimodal Deep Learning',
          contributor: 'Vinay Rautulwar',
          status: 'operational',
          modality: 'Text Tokens + 13 Scaled Behavioral Indicators',
          architecture: 'DeBERTa-v3 768d + 13d Dense Fusion + Focal Loss',
          checkpoint: 'backend/models/hybrid_deberta_tabular/',
          dataset: '26,844 reviews (Zero Author/Template Leakage)',
          metrics: 'ROC-AUC: 0.7472 (Top) • PR-AUC: 0.7002 • Recall: 66.73%',
          active: true,
        },
        {
          name: 'TF-IDF + LightGBM Baseline',
          id: 'baseline',
          category: 'Gradient Boosted Decision Trees',
          contributor: 'Vinay Rautulwar',
          status: 'operational',
          modality: 'Word N-Grams (1,2) TF-IDF (5,000 features)',
          architecture: 'scikit-learn TF-IDF + LightGBM 4.0 Trees',
          checkpoint: 'training/artifacts/baseline/',
          dataset: '17,896 Train / 4,472 Held-Out Validation',
          metrics: 'ROC-AUC: 0.6691 • PR-AUC: 0.6118 • Inference: <5ms',
          active: true,
        },
        {
          name: 'Packed Bidirectional LSTM (BiLSTM)',
          id: 'bilstm',
          category: 'Recurrent Sequence Modeling',
          contributor: 'Pramit Ghosh',
          status: 'operational',
          modality: 'Token Sequences (1,090 word vocabulary)',
          architecture: 'PyTorch PackedSequence (Embedding 128d, Hidden 64d)',
          checkpoint: 'training/artifacts/bilstm/bilstm_model.pt',
          dataset: '4,474 Held-Out Test Reviews',
          metrics: 'ROC-AUC: 0.7157 • PR-AUC: 0.6402 • F1: 52.60%',
          active: true,
        },
        {
          name: 'Calibrated Linear SVM',
          id: 'svm',
          category: 'Maximum-Margin Linear Classifier',
          contributor: 'Pramit Ghosh',
          status: 'operational',
          modality: 'TF-IDF Unigrams & Bigrams',
          architecture: 'scikit-learn LinearSVC + CalibratedClassifierCV',
          checkpoint: 'training/artifacts/svm/svm_model.joblib',
          dataset: 'Ott 5-Fold Cross-Validation & Test Split',
          metrics: 'ROC-AUC: 0.6210 • Accuracy: 64.51% • Precision: 56.63%',
          active: true,
        },
        {
          name: 'FakeGuard Multi-View Cross-Attention Net',
          id: 'fakeguard',
          category: 'Multi-View Graph & Attention Architecture',
          contributor: 'Pramit Ghosh',
          status: 'experimental',
          modality: 'Text Embeddings + Behavioral Features + Relational GAT',
          architecture: 'Tri-View Multi-Head Cross-Attention Network',
          checkpoint: 'src/models/fakeguard.py (Experimental Weights)',
          dataset: 'Cross-Attention Multimodal Synthesizer',
          metrics: 'Outputs Explainable View Attention Weights Matrix',
          active: true,
        },
        {
          name: 'TF-IDF + Logistic Regression',
          id: 'lr_tfidf',
          category: 'Generalized Linear Model Benchmark',
          contributor: 'Pramit Ghosh',
          status: 'benchmark',
          modality: 'TF-IDF Text Vectors',
          architecture: 'L2-regularized Logistic Regression (C=1.0)',
          checkpoint: 'Saved in reports/unified_model_benchmark.json',
          dataset: '4,474 Held-Out Test Reviews',
          metrics: 'ROC-AUC: 0.7088 • PR-AUC: 0.6276 • F1: 53.99%',
          active: false,
        },
        {
          name: 'TF-IDF + Multinomial Naive Bayes',
          id: 'nb_tfidf',
          category: 'Generative Probabilistic Classifier',
          contributor: 'Pramit Ghosh',
          status: 'benchmark',
          modality: 'TF-IDF Text Vectors',
          architecture: 'MultinomialNB with Laplace Smoothing (alpha=1.0)',
          checkpoint: 'Saved in reports/unified_model_benchmark.json',
          dataset: '4,474 Held-Out Test Reviews',
          metrics: 'ROC-AUC: 0.6558 • PR-AUC: 0.5489 • F1: 46.50%',
          active: false,
        },
        {
          name: 'Behavioral Random Forest (13 Tabular Features)',
          id: 'rf_behavioral',
          category: 'Tabular Ensemble Benchmark',
          contributor: 'Pramit Ghosh',
          status: 'benchmark',
          modality: '13 Scaled Behavioral & Product Indicators',
          architecture: 'Random Forest Classifier (n_estimators=100)',
          checkpoint: 'Saved in reports/unified_model_benchmark.json',
          dataset: '4,474 Held-Out Test Reviews',
          metrics: 'ROC-AUC: 0.5701 • PR-AUC: 0.4491 • F1: 37.47%',
          active: false,
        },
        {
          name: 'Combined Logistic Regression (Text + Behavior)',
          id: 'lr_combined',
          category: 'Early Fusion Linear Baseline',
          contributor: 'Pramit Ghosh',
          status: 'benchmark',
          modality: 'TF-IDF Text Vectors + 13 Standardized Metadata Features',
          architecture: 'LogisticRegression with Feature Concatenation',
          checkpoint: 'Saved in reports/unified_model_benchmark.json',
          dataset: '4,474 Held-Out Test Reviews',
          metrics: 'ROC-AUC: 0.7085 • PR-AUC: 0.6273 • F1: 54.09%',
          active: false,
        },
        {
          name: 'Ott 5-Fold Stratified Ensemble',
          id: 'ott_ensemble',
          category: 'Leakage-Protected Cross-Validation',
          contributor: 'Shared Benchmark',
          status: 'benchmark',
          modality: 'Reviewer-Disjoint Stratified Splits',
          architecture: '5-Fold Cross-Validation Framework',
          checkpoint: 'results/baseline_results.json',
          dataset: 'Identical reviewer-isolated cross-folds',
          metrics: 'Mean Cross-Validation ROC-AUC: 0.6720 ± 0.008',
          active: false,
        },
        {
          name: 'Heuristic Rule-Based Fallback / Demo Mode',
          id: 'heuristic_demo',
          category: 'Non-ML Fallback & Watermarked Inspection',
          contributor: 'System Runtime',
          status: 'benchmark',
          modality: 'Keywords, Sentiment Polarities, and Purchase Flags',
          architecture: 'Deterministic Transparent Heuristics',
          checkpoint: 'frontend/js/api.js (simulateDemoPrediction)',
          dataset: 'Client-Side Disclaimed Watermarked Preview',
          metrics: 'Strictly Watermarked for Diagnostic Demonstrations',
          active: false,
        },
      ];

      container.innerHTML = '';
      catalog.forEach(item => {
        const card = document.createElement('div');
        card.className = 'registry-card';

        const badgeClass = item.status === 'operational' ? 'operational' : (item.status === 'experimental' ? 'experimental' : 'benchmark');
        const badgeLabel = item.status === 'operational' ? 'Operational' : (item.status === 'experimental' ? 'Experimental' : 'Benchmark');

        card.innerHTML = `
          <div class="registry-card-header">
            <div>
              <div class="registry-card-title">${this.escapeHtml(item.name)}</div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">${this.escapeHtml(item.category)}</div>
            </div>
            <span class="registry-badge ${badgeClass}">${badgeLabel}</span>
          </div>

          <div class="registry-meta-list">
            <div class="registry-meta-row">
              <span class="registry-meta-k">Contributor</span>
              <span class="registry-meta-v" style="color: var(--cyan-400);">${this.escapeHtml(item.contributor)}</span>
            </div>
            <div class="registry-meta-row">
              <span class="registry-meta-k">Modality</span>
              <span class="registry-meta-v">${this.escapeHtml(item.modality)}</span>
            </div>
            <div class="registry-meta-row">
              <span class="registry-meta-k">Architecture</span>
              <span class="registry-meta-v font-mono">${this.escapeHtml(item.architecture)}</span>
            </div>
            <div class="registry-meta-row">
              <span class="registry-meta-k">Artifact Checkpoint</span>
              <span class="registry-meta-v font-mono" style="font-size: 0.72rem;">${this.escapeHtml(item.checkpoint)}</span>
            </div>
            <div class="registry-meta-row" style="margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid var(--border-subtle);">
              <span class="registry-meta-k">Benchmark Performance</span>
              <span class="registry-meta-v" style="color: var(--emerald-400); font-weight: 600;">${this.escapeHtml(item.metrics)}</span>
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    } catch (err) {
      console.warn('Model registry rendering error:', err);
    }
  }

  /* ============================================================================
     Analysis History
     ============================================================================ */
  bindHistoryControls() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const refreshBtn = document.getElementById('refresh-history-btn');
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');

    filterButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.state.historyFilter = btn.getAttribute('data-class');
        this.state.historyPage = 1;
        this.loadHistory();
      });
    });

    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.loadHistory());
    }

    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        if (this.state.historyPage > 1) {
          this.state.historyPage -= 1;
          this.loadHistory();
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (this.state.historyPage < this.state.historyTotalPages) {
          this.state.historyPage += 1;
          this.loadHistory();
        }
      });
    }
  }

  async loadHistory() {
    const tbody = document.getElementById('history-tbody');
    const emptyState = document.getElementById('history-empty-state');
    const totalBadge = document.getElementById('history-total-badge');
    const pageInfo = document.getElementById('pagination-info');
    const prevBtn = document.getElementById('prev-page-btn');
    const nextBtn = document.getElementById('next-page-btn');

    try {
      const data = await window.api.getHistory({
        page: this.state.historyPage,
        pageSize: 15,
        predictionClass: this.state.historyFilter,
      });

      this.state.historyTotalPages = data.total_pages || 1;

      if (totalBadge) totalBadge.textContent = `${data.total.toLocaleString()} Records Stored`;
      if (pageInfo) pageInfo.textContent = `Page ${data.page} of ${this.state.historyTotalPages}`;

      if (prevBtn) prevBtn.disabled = data.page <= 1;
      if (nextBtn) nextBtn.disabled = data.page >= this.state.historyTotalPages;

      if (!tbody) return;
      tbody.innerHTML = '';

      if (data.items.length === 0) {
        if (emptyState) emptyState.style.display = 'flex';
        return;
      }

      if (emptyState) emptyState.style.display = 'none';

      data.items.forEach(item => {
        const tr = document.createElement('tr');
        const isFake = item.predicted_label === 'fake';
        const shortId = item.id ? item.id.substring(item.id.length - 8) : '—';
        const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : '—';

        tr.innerHTML = `
          <td class="font-mono" style="color: var(--cyan-400);">${shortId}</td>
          <td style="max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ${this.escapeHtml(item.review_text)}
          </td>
          <td>
            <span class="table-badge ${isFake ? 'fake' : 'genuine'}">${isFake ? 'Deceptive' : 'Genuine'}</span>
          </td>
          <td class="font-mono">${(item.fake_probability * 100).toFixed(1)}%</td>
          <td class="font-mono">${(item.confidence * 100).toFixed(1)}%</td>
          <td class="font-mono" style="font-size: 0.78rem;">${this.escapeHtml(item.model_name || '—')}</td>
          <td style="font-size: 0.78rem; color: var(--text-muted);">${dateStr}</td>
        `;

        tr.addEventListener('click', () => {
          this.openDetailModal(item);
        });

        tbody.appendChild(tr);
      });
    } catch (err) {
      console.warn('History fetch notice:', err.message);
      if (emptyState) emptyState.style.display = 'flex';
    }
  }

  /* ============================================================================
     Modals: Prediction Detail & Figure Zoom
     ============================================================================ */
  bindModal() {
    const modal = document.getElementById('prediction-detail-modal');
    const closeBtn = document.getElementById('close-modal-btn');

    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.remove('open'));
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('open');
      });
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('open')) {
          modal.classList.remove('open');
        }
      });
    }
  }

  openDetailModal(item) {
    const modal = document.getElementById('prediction-detail-modal');
    if (!modal) return;

    const isFake = item.predicted_label === 'fake';
    const badge = document.getElementById('modal-badge');
    const textEl = document.getElementById('modal-review-text');
    const idEl = document.getElementById('modal-id');
    const fakeEl = document.getElementById('modal-fake-prob');
    const confEl = document.getElementById('modal-confidence');
    const modelEl = document.getElementById('modal-model-name');
    const timeEl = document.getElementById('modal-timestamp');
    const metaSec = document.getElementById('modal-metadata-section');
    const metaJson = document.getElementById('modal-metadata-json');

    if (badge) {
      badge.className = `table-badge ${isFake ? 'fake' : 'genuine'}`;
      badge.textContent = isFake ? 'Deceptive' : 'Genuine';
    }
    if (textEl) textEl.textContent = item.review_text;
    if (idEl) idEl.textContent = item.id;
    if (fakeEl) fakeEl.textContent = (item.fake_probability * 100).toFixed(2) + '%';
    if (confEl) confEl.textContent = (item.confidence * 100).toFixed(2) + '%';
    if (modelEl) modelEl.textContent = item.model_name;
    if (timeEl) timeEl.textContent = new Date(item.timestamp).toUTCString();

    if (item.metadata && Object.keys(item.metadata).length > 0) {
      if (metaSec) metaSec.style.display = 'block';
      if (metaJson) metaJson.textContent = JSON.stringify(item.metadata, null, 2);
    } else {
      if (metaSec) metaSec.style.display = 'none';
    }

    modal.classList.add('open');
  }

  bindFigureModal() {
    const modal = document.getElementById('figure-zoom-modal');
    const closeBtn = document.getElementById('close-figure-modal-btn');

    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.remove('open'));
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('open');
      });
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('open')) {
          modal.classList.remove('open');
        }
      });
    }

    // Multi-port and relative fallback handler for figure images
    const figureImgs = document.querySelectorAll('.figure-img-wrap img, img.figure-img');
    figureImgs.forEach(img => {
      img.addEventListener('error', () => {
        const currentSrc = img.getAttribute('src');
        if (!img.dataset.retried) {
          img.dataset.retried = '1';
          const cleanPath = currentSrc.startsWith('/') ? currentSrc : '/' + currentSrc;
          img.src = 'http://127.0.0.1:8000' + cleanPath;
        }
      });
    });
  }

  openFigureModal(imgSrc, title) {
    const modal = document.getElementById('figure-zoom-modal');
    const img = document.getElementById('figure-modal-img');
    const titleEl = document.getElementById('figure-modal-title');

    if (modal && img) {
      img.dataset.modalRetried = '';
      img.src = imgSrc;
      img.onerror = () => {
        if (!img.dataset.modalRetried) {
          img.dataset.modalRetried = '1';
          const cleanPath = imgSrc.startsWith('/') ? imgSrc : '/' + imgSrc;
          img.src = 'http://127.0.0.1:8000' + cleanPath;
        }
      };
      if (titleEl) titleEl.textContent = title;
      modal.classList.add('open');
    }
  }

  escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Instantiate on window load
document.addEventListener('DOMContentLoaded', () => {
  window.app = new AppController();
});
