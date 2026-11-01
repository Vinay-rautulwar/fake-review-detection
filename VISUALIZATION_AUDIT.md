# Research Visualization Audit Report

**Project**: Fake Review Detection System (v2.0.0-integrated)  
**Workspace**: `fake-review-detection-v2`  
**Date of Audit**: November 1, 2026  
**Audit Purpose**: Complete audit of existing figures, identification of simulated/placeholder scripts, remediation of visualization workflows, and generation of genuine empirical figures for research publication.

---

## Executive Summary

Prior to this audit, several figures and scripts in the integrated repository relied on simulated data, random number generators, or synthetic features:
1. **Legacy Figures in `results/figures/`**: Generated during exploratory data analysis (EDA) using synthetic sample reviews generated because `dataset.csv` was missing from the originating repository.
2. **Simulation in Visualization Scripts**: `scripts/generate_figures.py` and `scripts/evaluate.py` generated predictions using `np.random.randint(0, 2, 1000)` and `np.random.randn(1000)` rather than loading model checkpoints.
3. **Data Leakage in Synthetic Metadata**: `scripts/train_real_datasets.py` generated behavioral metadata conditioned directly on `df['label'] == 1`.

### Actions Taken in this Remediation:
- **Removed Fake Logic**: `scripts/generate_figures.py` and `scripts/evaluate.py` have been refactored to remove all random/placeholder simulation logic. They now strictly ingest empirical test data from `data/splits/test.csv` (N = 4,474) and load verified model artifacts.
- **Generated Genuine Publication Figures**: 8 publication-quality figures (both 300 DPI PNG and vector PDF) were generated from verified model predictions on the identical 4,474 held-out test reviews and saved to `results/figures/research_paper/`.
- **Integrity Compliance (Rule 7)**: No missing predictions or training loss histories were fabricated. Because historical training runs of DeBERTa-v3 and BiLSTM did not record epoch-by-epoch loss arrays to disk, training loss curves are explicitly omitted rather than synthesized.

---

## 1. Audit of Existing Legacy Figures (`results/figures/`)

The 8 legacy figures in `results/figures/` originated from Pramit Ghosh's exploratory data analysis notebook (`notebooks/01_eda.py`).

| Figure File | Display Title / Content | Data Source | Sample Size | Valid for Research? | Defect / Reason |
|---|---|---|---|---|---|
| `eda_class_distribution.png` / `.pdf` | Class Distribution (Genuine vs Fake) | Synthetic sample generation (`generate_sample_dataset()`) | 3,500 reviews | **NO** | Raw data was missing in originating repo; script fell back to generating synthetic reviews with hardcoded class ratio. |
| `eda_review_length_distribution.png` | Review Length Distribution | Synthetic review text | 3,500 reviews | **NO** | Text lengths reflect synthetic string generation template, not real consumer reviews. |
| `eda_rating_distribution.png` | Star Rating Distribution | Synthetic ratings | 3,500 reviews | **NO** | Star ratings generated from synthetic distribution parameters. |
| `eda_wordclouds.png` | Word Clouds for Genuine & Fake Reviews | Synthetic text strings | 3,500 reviews | **NO** | Words reflect placeholder synthetic vocabulary. |
| `eda_top_features_comparison.png` | Standardized Differences (Cohen's d) | Synthetic behavioral features | 3,500 reviews | **NO** | Behavioral columns were synthesized directly using Gaussian distributions conditioned on label. |
| `eda_behavioral_correlation_heatmap.png` / `.pdf` | Behavioral Features Correlation Matrix | Pearson correlation on synthetic features | 3,500 reviews | **NO** | Correlations reflect synthetic generator covariance rather than genuine reviewer behavior. |

**Verdict on Legacy Figures**: None of the legacy `results/figures/eda_*` figures are valid for peer-reviewed research publication. They must be cited only as exploratory artifacts or replaced with the genuine figures generated in `results/figures/research_paper/`.

---

## 2. Audit of Legacy Visualization & Evaluation Scripts

| Script File | Legacy Implementation Details | Identified Defect | Remediation Status |
|---|---|---|---|
| `scripts/generate_figures.py` | Lines 31–41, 50–55: `y_true = np.random.randint(0, 2, 1000)`, `prob_fg = np.clip(y_true * 0.85 + np.random.randn(1000) * 0.2, 0, 1)` | Generates paper ROC curves, ablation bar charts, and LaTeX tables from numpy random numbers. | **REMEDIATED**: Completely refactored. Now imports genuine predictions from `scripts/generate_publication_figures.py` and evaluates 4,474 real reviews. |
| `scripts/evaluate.py` | Lines 33–36: `y_true = np.random.randint(0, 2, 1000)`, `y_prob = np.clip(y_true * 0.8 + np.random.randn(1000) * 0.3, 0, 1)` | Does not load model checkpoint; computes metrics and confusion matrix on 1,000 random integers. | **REMEDIATED**: Completely refactored. Ingests `data/splits/test.csv`, loads cached/model predictions, and runs `MetricsCalculator` on real reviews. |
| `scripts/train_real_datasets.py` | Lines 144–180: `_synthesize_behavioral_features()` with `fake_mask = df['label'] == 1` | Synthesizes behavioral features (`rating_deviation`, `extreme_rating`, etc.) directly conditioned on the label. | **IDENTIFIED**: Documented as severe research leakage defect in `POST_INTEGRATION_AUDIT.md`. Must not be used for multimodal claims. |

---

## 3. Genuine Publication-Quality Figures (`results/figures/research_paper/`)

All figures below have been generated using `scripts/generate_publication_figures.py` and saved in both **high-resolution PNG (300 DPI)** and **vector PDF** format in `results/figures/research_paper/`.

Every model comparison in Figures 1–5 uses the **exact same 4,474 test reviews** from `data/splits/test.csv` (2,797 Genuine, 1,677 Fake).

### Figure 1: Model Performance Comparison Bar Chart
- **Files**: `fig1_model_performance_comparison.png`, `fig1_model_performance_comparison.pdf`
- **Data Source**: Real test predictions for 7 verified model configurations on `data/splits/test.csv` (N = 4,474).
- **Metrics Plotted**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- **Threshold**: Fixed default decision threshold $\tau = 0.50$ across all models for fair comparison.
- **Key Insight**: DeBERTa-v3 Text-Only achieves the highest precision (86.74%), while DeBERTaTabularFusionNet (Class-Weighted) achieves the highest F1-Score (60.41%) and Recall (81.40%).
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 2: Combined ROC Curves
- **Files**: `fig2_combined_roc_curves.png`, `fig2_combined_roc_curves.pdf`
- **Data Source**: Continuous predicted probabilities on identical 4,474 test reviews.
- **Models Overlaid**:
  1. DeBERTaTabularFusionNet (Focal Loss) — **AUC: 0.7472**
  2. DeBERTaTabularFusionNet (Class-Weighted) — **AUC: 0.7459**
  3. LightGBM Multimodal Hybrid — **AUC: 0.7425**
  4. Packed BiLSTM — **AUC: 0.7157**
  5. TF-IDF + LightGBM Baseline — **AUC: 0.7157**
  6. DeBERTa-v3-base (Text-Only) — **AUC: 0.7152**
  7. Calibrated Linear SVM — **AUC: 0.6210**
  8. Random Chance Baseline — **AUC: 0.5000**
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 3: Combined Precision-Recall Curves
- **Files**: `fig3_combined_precision_recall_curves.png`, `fig3_combined_precision_recall_curves.pdf`
- **Data Source**: Continuous predicted probabilities evaluated against true deceptive review prevalence.
- **Prevalence Baseline**: Horizontal dashed line at $1,677 / 4,474 = 0.3748$ (test class prevalence).
- **PR-AUC (Average Precision) Scores**:
  1. DeBERTaTabularFusionNet (Focal Loss) — **PR-AUC: 0.7002**
  2. DeBERTaTabularFusionNet (Class-Weighted) — **PR-AUC: 0.7000**
  3. LightGBM Multimodal Hybrid — **PR-AUC: 0.6953**
  4. DeBERTa-v3-base (Text-Only) — **PR-AUC: 0.6639**
  5. TF-IDF + LightGBM Baseline — **PR-AUC: 0.6624**
  6. Packed BiLSTM — **PR-AUC: 0.6583**
  7. Calibrated Linear SVM — **PR-AUC: 0.5167**
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 4: Multi-Model Confusion Matrix Grid
- **Files**: `fig4_confusion_matrices.png`, `fig4_confusion_matrices.pdf`
- **Data Source**: Actual test predictions on 4,474 reviews. Displays both raw counts (TN, FP, FN, TP) and row-normalized class percentages.
- **Models Included**:
  1. DeBERTa-v3 Text (Default, $\tau=0.50$): TN=2,725 (97.4%), FP=72 (2.6%), FN=1,206 (71.9%), TP=471 (28.1%).
  2. DeBERTa-v3 Text (Val-Tuned, $\tau=0.12$): TN=883 (31.6%), FP=1,914 (68.4%), FN=131 (7.8%), TP=1,546 (92.2%).
  3. TF-IDF + LightGBM (Default, $\tau=0.50$): TN=1,862 (66.6%), FP=934 (33.4%), FN=781 (46.6%), TP=895 (53.4%).
  4. Packed BiLSTM (Default, $\tau=0.50$): TN=2,162 (77.3%), FP=635 (22.7%), FN=852 (50.8%), TP=825 (49.2%).
  5. Linear SVM (Default, $\tau=0.50$): TN=2,676 (95.7%), FP=121 (4.3%), FN=1,467 (87.5%), TP=210 (12.5%).
  6. FusionNet Focal Loss (Default, $\tau=0.50$): TN=1,811 (64.7%), FP=986 (35.3%), FN=558 (33.3%), TP=1,119 (66.7%).
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 5: Decision Threshold Sensitivity Analysis
- **Files**: `fig5_threshold_sensitivity_analysis.png`, `fig5_threshold_sensitivity_analysis.pdf`
- **Data Source**: Systematic sweep of decision thresholds $\tau \in [0.02, 0.98]$ on test set predictions.
- **Subplots**:
  - Subplot A: DeBERTa-v3 Text-Only (showing trade-off between Precision, Recall, and F1, highlighting default $\tau=0.50$ vs. validation-derived optimal $\tau=0.12$).
  - Subplot B: DeBERTaTabularFusionNet Focal Loss (highlighting default $\tau=0.50$ vs. validation-derived optimal $\tau=0.41$).
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 6: Behavioral Feature Correlation Matrix
- **Files**: `fig6_behavioral_correlation_heatmap.png`, `fig6_behavioral_correlation_heatmap.pdf`
- **Data Source**: 13 engineered tabular metadata features extracted from real e-commerce reviews in `data/splits/test.csv`.
- **Features Included**:
  `rating`, `product_avg_rating`, `rating_discrepancy`, `abs_rating_discrepancy`, `reviewer_avg_rating`, `reviewer_rating_discrepancy`, `verified_purchase`, `reviewer_review_count`, `days_since_previous_review`, `reviewer_account_age_days`, `reviewer_verified_ratio`, `product_price`, `product_review_count`.
- **Styling**: Half-matrix (upper triangle masked), diverging `vlag` palette, Pearson $r$ annotated in each cell.
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 7: LightGBM Feature Importance (Baseline & Hybrid)
- **Files**: `fig7_feature_importance_lightgbm.png`, `fig7_feature_importance_lightgbm.pdf`
- **Data Source**: Split feature importances extracted directly from trained LightGBM models (`lightgbm_model.joblib` and `lgb_hybrid_standard.joblib`).
- **Subplots**:
  - Subplot A: Baseline top 15 predictive TF-IDF n-grams (e.g., promotional terms, generic praise).
  - Subplot B: Multimodal Hybrid feature importances, showing the relative ranking of DeBERTa contextual text representation versus engineered behavioral metadata features.
- **Valid for Publication**: **YES (100% Empirical)**.

### Figure 8: Standardized Behavioral Effect Sizes (Cohen's d)
- **Files**: `fig8_behavioral_standardized_differences.png`, `fig8_behavioral_standardized_differences.pdf`
- **Data Source**: Cohen's $d = (\mu_{\text{fake}} - \mu_{\text{genuine}}) / s_{\text{pooled}}$ calculated across all 13 tabular features on the 4,474 real test reviews.
- **Color Coding**: Orange for features elevated in deceptive reviews; Blue for features elevated in authentic reviews.
- **Valid for Publication**: **YES (100% Empirical)**.

---

## 4. Audit of Training and Validation Loss Curves

### Policy Check & Compliance with Rule 7
- **User Instruction 7**: *"Never fabricate missing predictions or training history."*
- **User Instruction 5**: *"Training and validation loss curves, if logs are available"*

### Audit Findings:
1. Checked `backend/models/deberta_fp32_final/`: Contains `model.safetensors`, `config.json`, `tokenizer.json`, and `validation_metrics.json`. No `trainer_state.json` or per-step loss log was saved.
2. Checked `training/artifacts/bilstm/`: Contains `best_bilstm.pt` and `bilstm_metrics.json`. Only final epoch metrics were persisted; step-level loss arrays were not written to disk.
3. Checked `backend/models/hybrid_deberta_tabular/`: Contains checkpoint state dicts and `benchmark_comparison.json`.

### Conclusion:
Because epoch-by-epoch training/validation loss logs were not persisted to disk during model training, **loss curves are intentionally omitted rather than synthesized from mock mathematical functions**. Fabricating artificial loss curves would violate scientific ethics and the explicit constraints of this audit.

---

## 5. Summary Table: Figure Validity for Research Papers

| Figure Number | Filename (in `results/figures/research_paper/`) | Modality | Sample Size ($N$) | Ground Truth | Valid for Publication? |
|---|---|---|---|---|---|
| **Figure 1** | `fig1_model_performance_comparison` | Multimodal & Text | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 2** | `fig2_combined_roc_curves` | Multimodal & Text | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 3** | `fig3_combined_precision_recall_curves` | Multimodal & Text | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 4** | `fig4_confusion_matrices` | Multimodal & Text | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 5** | `fig5_threshold_sensitivity_analysis` | Multimodal & Text | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 6** | `fig6_behavioral_correlation_heatmap` | Tabular Metadata | 4,474 reviews | `data/splits/test.csv` | **VALID** |
| **Figure 7** | `fig7_feature_importance_lightgbm` | Text & Tabular | 4,474 reviews | Model Trees | **VALID** |
| **Figure 8** | `fig8_behavioral_standardized_differences` | Tabular Metadata | 4,474 reviews | `data/splits/test.csv` | **VALID** |
