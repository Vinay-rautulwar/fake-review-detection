/**
 * Fake Review Detection v2 - Application Controller
 * Handles UI interactions, routing, live polling, threshold switching,
 * model lifecycle detection, and transparent simulation watermark handling.
 */

class AppController {
  constructor() {
    this.state = {
      activeTab: 'overview',
      systemHealth: null,
      modelsList: [],
      activeModel: null,
      analyticsSummary: null,
      modelMetrics: null,
      currentThreshold: '0.5',
      historyPage: 1,
      historyFilter: 'all',
      historyTotalPages: 1,
      batchResults: [],
    };

    // Sample reviews for quick-fill
    this.sampleReviews = {
      'suspicious-promo': {
        text: 'This product is absolutely incredible and exceeded every single one of my expectations! Flawless build, arrived in 1 day, changed my life! Highly recommend to everyone, must buy immediately!',
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
    };

    this.init();
  }

  /**
   * Initialize DOM event listeners and load real-time system state.
   */
  async init() {
    this.bindNavigation();
    this.bindAnalyzerForm();
    this.bindEvaluationView();
    this.bindBatchScanner();
    this.bindHistoryControls();
    this.bindModal();

    // Initial data fetch
    await this.refreshSystemStatus();
    await this.loadAnalytics();
    await this.loadMetrics();
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
      btn.addEventListener('click', (e) => {
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
    }
  }

  /* ============================================================================
     System Status & Model Lifecycle (Requirements 7 & 8)
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
      this.updateModelContextBanner(health, modelsData);
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

    // 2. Model Badge (Requirement 8: Training, Model not loaded, or Ready)
    const activeModelMeta = this.state.modelsList.find(m => m.name === modelsData.active_model);
    const isAvailable = activeModelMeta ? activeModelMeta.is_available : false;
    const isLoaded = activeModelMeta ? activeModelMeta.is_loaded : false;
    const modelName = modelsData.active_model || 'Unknown';

    if (isAvailable && isLoaded) {
      modelPill.className = 'status-pill ready';
      modelText.textContent = `Model: Ready (${modelName})`;
    } else if (modelName.includes('deberta')) {
      modelPill.className = 'status-pill training';
      modelText.textContent = 'Model: Training in Colab';
    } else if (activeModelMeta && !activeModelMeta.is_loaded) {
      modelPill.className = 'status-pill not-loaded';
      modelText.textContent = 'Model: Not Loaded';
    } else {
      modelPill.className = 'status-pill unavailable';
      modelText.textContent = 'Model: Unavailable';
    }
  }

  updateOverviewModelSpecs(modelsData) {
    const activeKey = modelsData.active_model;
    const modelMeta = this.state.modelsList.find(m => m.name === activeKey) || {
      name: activeKey,
      architecture: 'DeBERTa-v3 Contextual Embeddings + Dense Fusion',
      input_type: 'multimodal',
      artifact_path: `training/artifacts/${activeKey}`,
      is_loaded: false,
      is_available: false,
      status_message: 'Training in progress in Google Colab.',
    };

    const nameEl = document.getElementById('spec-model-name');
    const archEl = document.getElementById('spec-model-arch');
    const modalityEl = document.getElementById('spec-model-modality');
    const pathEl = document.getElementById('spec-model-path');
    const loadedEl = document.getElementById('spec-model-loaded');
    const statusDescEl = document.getElementById('spec-model-status-desc');
    const cardActiveName = document.getElementById('card-active-model-name');
    const cardStatusSub = document.getElementById('card-model-status-sub');
    const inputBadge = document.getElementById('badge-model-input');

    if (nameEl) nameEl.textContent = modelMeta.name;
    if (archEl) archEl.textContent = modelMeta.architecture;
    if (modalityEl) modalityEl.textContent = modelMeta.input_type === 'multimodal' ? 'Text + Tabular Behavioral Metadata' : 'Raw Review Text Only';
    if (pathEl) pathEl.textContent = modelMeta.artifact_path;
    if (loadedEl) {
      loadedEl.textContent = modelMeta.is_loaded ? 'Loaded in Memory (Fast Live Inference)' : 'Not Loaded';
      loadedEl.style.color = modelMeta.is_loaded ? 'var(--emerald-400)' : 'var(--text-secondary)';
    }
    if (statusDescEl) {
      statusDescEl.textContent = modelMeta.status_message;
      statusDescEl.style.color = modelMeta.is_available ? 'var(--emerald-400)' : 'var(--amber-400)';
    }
    if (cardActiveName) cardActiveName.textContent = modelMeta.display_name || modelMeta.name;
    if (cardStatusSub) {
      cardStatusSub.textContent = modelMeta.is_available ? 'Ready for Live Inference' : 'Training Pending (Colab GPU)';
    }
    if (inputBadge) inputBadge.textContent = modelMeta.input_type;
  }

  updateModelContextBanner(health, modelsData) {
    const banner = document.getElementById('model-context-banner');
    const title = document.getElementById('banner-title');
    const msg = document.getElementById('banner-message');

    if (!banner) return;

    const activeKey = modelsData.active_model;
    const isModelAvailable = health.model && health.model.status === 'available';

    if (activeKey.includes('deberta') && !isModelAvailable) {
      banner.style.display = 'flex';
      banner.className = 'banner-alert warning';
      title.textContent = 'Colab GPU Training Mode Active';
      msg.innerHTML = `
        The active model is configured as <strong>${activeKey}</strong>, which is currently training separately in Google Colab (Step 6).
        The trained <strong>TF-IDF + LightGBM Baseline</strong> model is already trained and available on disk.
        You can classify reviews using the Baseline model by setting <code class="font-mono">MODEL_TYPE=baseline</code> in <code class="font-mono">.env</code>, or explore with watermarked simulation mode.
      `;
    } else {
      banner.style.display = 'none';
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
      console.log('Notice: MongoDB predictions count query returned:', err.message);
    }
  }

  /* ============================================================================
     Review Analyzer (Requirement 3 & 7)
     ============================================================================ */
  bindAnalyzerForm() {
    const form = document.getElementById('review-analyzer-form');
    const textarea = document.getElementById('review-text-input');
    const charCounter = document.getElementById('review-char-count');
    const clearBtn = document.getElementById('clear-input-btn');
    const accordionBtn = document.getElementById('meta-accordion-btn');
    const accordionContent = document.getElementById('meta-accordion-content');
    const sampleButtons = document.querySelectorAll('.sample-btn');
    const demoModeBtn = document.getElementById('try-demo-mode-btn');

    // Live character counter
    if (textarea && charCounter) {
      textarea.addEventListener('input', () => {
        const count = textarea.value.trim().length;
        charCounter.textContent = `${count} char${count === 1 ? '' : 's'}`;
        charCounter.classList.toggle('valid', count >= 3);
      });
    }

    // Accordion toggle
    if (accordionBtn && accordionContent) {
      accordionBtn.addEventListener('click', () => {
        const isOpen = accordionContent.classList.toggle('open');
        accordionBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
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
            accordionBtn.setAttribute('aria-expanded', 'true');
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

    // Submit handler
    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.handleAnalysisSubmit();
      });
    }

    // Demo simulation mode trigger
    if (demoModeBtn) {
      demoModeBtn.addEventListener('click', () => {
        this.runSimulatedDemoAnalysis();
      });
    }
  }

  async handleAnalysisSubmit() {
    if (this.isSubmitting) return; // Prevent duplicate submissions

    const textarea = document.getElementById('review-text-input');
    const text = textarea ? textarea.value.trim() : '';

    if (!text || text.length < 3) {
      this.renderAnalyzerError({
        status: 400,
        message: 'Review content must contain at least 3 non-whitespace characters.',
      });
      return;
    }

    if (text.length > 10000) {
      this.renderAnalyzerError({
        status: 400,
        message: 'Review content exceeds maximum limit of 10,000 characters.',
      });
      return;
    }

    // Collect optional metadata
    const ratingVal = document.getElementById('meta-rating').value;
    const verifiedVal = document.getElementById('meta-verified').value;
    const categoryVal = document.getElementById('meta-category').value.trim();
    const priceVal = document.getElementById('meta-price').value;
    const reviewerCountVal = document.getElementById('meta-reviewer-count').value;
    const daysPrevVal = document.getElementById('meta-days-prev').value;

    const payload = {
      review_text: text,
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
      this.renderPredictionResult(result, false);
      // Refresh persistent analytics and history
      this.loadAnalytics();
      this.loadHistory();
    } catch (err) {
      console.warn('Backend inference call rejected:', err);
      // If HTTP 503 (model unavailable / training), render truthful error state
      this.renderAnalyzerError(err);
    } finally {
      this.isSubmitting = false;
      const submitBtn = document.getElementById('submit-analysis-btn');
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  runSimulatedDemoAnalysis() {
    const textarea = document.getElementById('review-text-input');
    const text = textarea ? textarea.value.trim() : 'Sample test review';

    const verifiedVal = document.getElementById('meta-verified').value;
    const reviewerCountVal = document.getElementById('meta-reviewer-count').value;

    const metadata = {
      verified_purchase: verifiedVal === 'true' ? true : (verifiedVal === 'false' ? false : undefined),
      reviewer_review_count: reviewerCountVal ? parseInt(reviewerCountVal, 10) : undefined,
    };

    const simResult = window.api.simulateDemoPrediction(text, metadata);
    this.renderPredictionResult(simResult, true);
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

    if (submitBtn) {
      submitBtn.disabled = (state === 'loading');
    }
  }

  renderPredictionResult(result, isDemo = false) {
    this.setAnalyzerState('result');

    const watermark = document.getElementById('demo-watermark-pill');
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const verdictIconSvg = document.getElementById('verdict-icon-svg');

    const confVal = document.getElementById('gauge-confidence-val');
    const confMeter = document.getElementById('meter-confidence-fill');
    const fakeVal = document.getElementById('gauge-fake-val');
    const fakeMeter = document.getElementById('meter-fake-fill');
    const genuineLabel = document.getElementById('gauge-genuine-label');

    const modelNameEl = document.getElementById('result-model-name');
    const timestampEl = document.getElementById('result-timestamp');
    const badgeEl = document.getElementById('result-model-badge');

    // Watermark display
    if (watermark) {
      watermark.style.display = isDemo ? 'flex' : 'none';
    }

    const isFake = (result.predicted_label === 'fake' || result.prediction === 'fake' || result.label === 1);
    const fakeProb = (result.fake_probability !== undefined) ? result.fake_probability : (isFake ? result.confidence : 1.0 - result.confidence);
    const genuineProb = (result.genuine_probability !== undefined) ? result.genuine_probability : (1.0 - fakeProb);
    const confidence = (result.confidence !== undefined) ? result.confidence : Math.max(fakeProb, genuineProb);

    const confPct = Math.round(confidence * 1000) / 10;
    const fakePct = Math.round(fakeProb * 1000) / 10;
    const genuinePct = Math.round(genuineProb * 1000) / 10;

    // Verdict Styling
    if (verdictBanner) {
      verdictBanner.className = `verdict-banner ${isFake ? 'fake' : 'genuine'}`;
    }
    if (verdictTitle) {
      verdictTitle.textContent = isFake ? 'DECEPTIVE / FAKE REVIEW' : 'AUTHENTIC / GENUINE REVIEW';
    }
    if (verdictDesc) {
      verdictDesc.textContent = isFake
        ? 'High probability of synthetic generator pattern or deceptive promotional intent.'
        : 'High consistency with authentic human reviewer sentiment and linguistic structure.';
    }
    if (verdictIconSvg) {
      verdictIconSvg.innerHTML = isFake
        ? '<path d="M18 6L6 18M6 6l12 12"></path>'
        : '<polyline points="20 6 9 17 4 12"></polyline>';
    }

    // Gauges
    if (confVal) confVal.textContent = `${confPct}%`;
    if (confMeter) confMeter.style.width = `${confPct}%`;
    if (genuineLabel) genuineLabel.textContent = `Genuine: ${genuinePct}%`;

    if (fakeVal) {
      fakeVal.textContent = `${fakePct}%`;
      fakeVal.style.color = isFake ? 'var(--rose-400)' : 'var(--emerald-400)';
    }
    if (fakeMeter) {
      fakeMeter.className = `meter-fill ${isFake ? 'rose' : 'emerald'}`;
      fakeMeter.style.width = `${fakePct}%`;
    }

    // Model and Source
    const activeModelName = result.model || result.model_name || 'deberta-v3-base';
    if (modelNameEl) modelNameEl.textContent = activeModelName;
    if (timestampEl) timestampEl.textContent = result.timestamp ? new Date(result.timestamp).toUTCString() : new Date().toUTCString();
    if (badgeEl) {
      badgeEl.textContent = isDemo ? 'SIMULATED DEMO' : 'LIVE DEBERTA INFERENCE';
      badgeEl.style.color = isDemo ? 'var(--amber-400)' : 'var(--emerald-400)';
    }
  }

  renderAnalyzerError(err) {
    this.setAnalyzerState('error');

    const titleEl = document.getElementById('error-state-title');
    const msgEl = document.getElementById('error-state-message');
    const badgeEl = document.getElementById('result-model-badge');

    if (titleEl) {
      titleEl.textContent = (err.status === 503) ? 'Model Training / Unavailable (HTTP 503)' : 'Inference Request Error';
    }
    if (msgEl) {
      msgEl.textContent = err.message || 'The backend cannot serve predictions with the current model configuration.';
    }
    if (badgeEl) {
      badgeEl.textContent = 'HTTP 503';
      badgeEl.style.color = 'var(--rose-400)';
    }
  }

  /* ============================================================================
     Model Evaluation & Benchmarks (Requirement 3 & 7)
     ============================================================================ */
  bindEvaluationView() {
    // Sub-tab switcher
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

    // Threshold switcher
    const thresholdPills = document.querySelectorAll('.switch-pill');
    thresholdPills.forEach(pill => {
      pill.addEventListener('click', () => {
        const t = pill.getAttribute('data-threshold');
        thresholdPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        this.updateEvaluationMetricsForThreshold(t);
      });
    });
  }

  async loadMetrics() {
    try {
      const data = await window.api.getModelMetrics();
      this.state.modelMetrics = data;
      this.updateEvaluationMetricsForThreshold(this.state.currentThreshold);
    } catch (err) {
      console.warn('Model metrics retrieval returned:', err);
    }
  }

  updateEvaluationMetricsForThreshold(thresholdKey) {
    this.state.currentThreshold = thresholdKey;

    const data = this.state.modelMetrics;
    const baseline = data ? data.baseline : null;
    const metrics = baseline ? baseline.metrics : null;

    const rocEl = document.getElementById('metric-roc-auc');
    const prEl = document.getElementById('metric-pr-auc');
    const precEl = document.getElementById('metric-precision');
    const recEl = document.getElementById('metric-recall');
    const f1El = document.getElementById('metric-f1');
    const badge = document.getElementById('cm-threshold-badge');

    const tnEl = document.getElementById('cm-tn-count');
    const fpEl = document.getElementById('cm-fp-count');
    const fnEl = document.getElementById('cm-fn-count');
    const tpEl = document.getElementById('cm-tp-count');

    if (!metrics) {
      // Offline fallback values from reports/baseline_validation_report.txt
      if (thresholdKey === '0.5') {
        if (rocEl) rocEl.textContent = '0.6691';
        if (prEl) prEl.textContent = '0.6118';
        if (precEl) precEl.textContent = '0.4893';
        if (recEl) recEl.textContent = '0.5340';
        if (f1El) f1El.textContent = '0.5107';
        if (badge) badge.textContent = 'Threshold = 0.50 (Default)';
        if (tnEl) tnEl.textContent = '1,862';
        if (fpEl) fpEl.textContent = '934';
        if (fnEl) fnEl.textContent = '781';
        if (tpEl) tpEl.textContent = '895';
      } else {
        if (rocEl) rocEl.textContent = '0.6691';
        if (prEl) prEl.textContent = '0.6118';
        if (precEl) precEl.textContent = '0.4259';
        if (recEl) recEl.textContent = '0.9433';
        if (f1El) f1El.textContent = '0.5869';
        if (badge) badge.textContent = 'Threshold = 0.25 (Validation-Optimal)';
        if (tnEl) tnEl.textContent = '665';
        if (fpEl) fpEl.textContent = '2,131';
        if (fnEl) fnEl.textContent = '95';
        if (tpEl) tpEl.textContent = '1,581';
      }
      return;
    }

    // Dynamic metrics from validation_metrics.json
    if (rocEl) rocEl.textContent = metrics.roc_auc.toFixed(4);
    if (prEl) prEl.textContent = metrics.pr_auc.toFixed(4);

    const tObj = (thresholdKey === '0.25' && metrics.optimal_threshold)
      ? metrics.optimal_threshold
      : (metrics.threshold_0_5 || metrics.optimal_threshold);

    if (tObj) {
      if (precEl) precEl.textContent = tObj.precision.toFixed(4);
      if (recEl) recEl.textContent = tObj.recall.toFixed(4);
      if (f1El) f1El.textContent = tObj.f1_score.toFixed(4);
      if (badge) badge.textContent = `Threshold = ${tObj.threshold.toFixed(2)}`;

      if (tObj.confusion_matrix) {
        if (tnEl) tnEl.textContent = tObj.confusion_matrix.tn.toLocaleString();
        if (fpEl) fpEl.textContent = tObj.confusion_matrix.fp.toLocaleString();
        if (fnEl) fnEl.textContent = tObj.confusion_matrix.fn.toLocaleString();
        if (tpEl) tpEl.textContent = tObj.confusion_matrix.tp.toLocaleString();
      }
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
    const resultsPanel = document.getElementById('batch-results-panel');

    if (progressBox) progressBox.style.display = 'block';
    if (fileNameEl) fileNameEl.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    if (statusEl) {
      statusEl.className = 'status-pill training';
      statusEl.textContent = 'Processing Batch';
    }

    try {
      const data = await window.api.predictBatch(file);
      this.state.batchResults = data.predictions || [];

      if (statusEl) {
        statusEl.className = 'status-pill ready';
        statusEl.textContent = 'Completed';
      }

      this.renderBatchResults(data);
      this.loadAnalytics();
      this.loadHistory();
    } catch (err) {
      console.warn('Batch processing failed:', err);
      if (statusEl) {
        statusEl.className = 'status-pill unavailable';
        statusEl.textContent = 'Failed';
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
    if (summaryBadge) {
      summaryBadge.textContent = `${data.total_successful} Success / ${data.total_failed} Failed of ${data.total_submitted}`;
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
            : '<span class="table-badge unavailable">N/A</span>'
          }
        </td>
        <td class="font-mono">${item.fake_probability !== null ? (item.fake_probability * 100).toFixed(1) + '%' : '—'}</td>
        <td class="font-mono">${item.confidence !== null ? (item.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td>
          <span style="font-size: 0.75rem; color: ${isSuccess ? 'var(--emerald-400)' : 'var(--rose-400)'};">
            ${isSuccess ? 'Success' : 'Error'}
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
     Analysis History (Requirement 3 & 4)
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
            <span class="table-badge ${isFake ? 'fake' : 'genuine'}">${isFake ? 'Fake' : 'Genuine'}</span>
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
     Detail Modal
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
      badge.textContent = isFake ? 'Fake' : 'Genuine';
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
