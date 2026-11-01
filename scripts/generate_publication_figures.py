"""
Generate publication-quality, empirical figures and tables for the Fake Review Detection project.
Strictly uses verified test predictions and actual dataset labels from data/splits/test.csv.
No simulated, random, or hardcoded mock predictions.

Outputs saved in results/figures/research_paper/ (both PNG and PDF):
1. fig1_model_performance_comparison (.png, .pdf)
2. fig2_combined_roc_curves (.png, .pdf)
3. fig3_combined_precision_recall_curves (.png, .pdf)
4. fig4_confusion_matrices (.png, .pdf)
5. fig5_threshold_sensitivity_analysis (.png, .pdf)
6. fig6_behavioral_correlation_heatmap (.png, .pdf)
7. fig7_feature_importance_lightgbm (.png, .pdf)
8. fig8_behavioral_standardized_differences (.png, .pdf)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.hybrid_fusion import DeBERTaTabularFusionNet, extract_tabular_features

# Set high-quality publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'semibold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'legend.title_fontsize': 10.5,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Distinct, professional color palette
MODEL_COLORS = {
    'TF-IDF + LightGBM Baseline': '#4c72b0',
    'Calibrated Linear SVM': '#dd8452',
    'Packed BiLSTM': '#55a868',
    'DeBERTa-v3-base (Text-Only)': '#c44e52',
    'LightGBM Multimodal Hybrid': '#8172b3',
    'DeBERTaTabularFusionNet (Class-Weighted)': '#937860',
    'DeBERTaTabularFusionNet (Focal Loss)': '#da8bc3',
}


def load_test_data_and_predictions(project_root: Path):
    """Load ground-truth test labels and actual model predictions."""
    test_csv = project_root / 'data' / 'splits' / 'test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found at {test_csv}")

    test_df = pd.read_csv(test_csv)
    y_true = (test_df['label'].astype(str).str.lower() == 'fake').astype(int).values
    total_reviews = len(y_true)
    fake_count = int(y_true.sum())
    genuine_count = total_reviews - fake_count
    print(f"Loaded test set: {total_reviews} reviews ({genuine_count} genuine, {fake_count} fake).")

    # 1. DeBERTa-v3 Text Probs
    deb_probs_path = project_root / 'training' / 'artifacts' / 'cache' / 'test_fake_probs.npy'
    if not deb_probs_path.exists():
        raise FileNotFoundError(f"DeBERTa predictions cache missing: {deb_probs_path}")
    deb_probs = np.load(deb_probs_path)

    # 2. Packed BiLSTM Probs
    bilstm_probs_path = project_root / 'training' / 'artifacts' / 'bilstm' / 'test_fake_probs.npy'
    bilstm_probs = np.load(bilstm_probs_path)

    # 3. Calibrated Linear SVM Probs
    svm_probs_path = project_root / 'training' / 'artifacts' / 'svm' / 'test_fake_probs.npy'
    svm_probs = np.load(svm_probs_path)

    # 4. TF-IDF + LightGBM Baseline Probs
    lgb_base_model = joblib.load(project_root / 'training' / 'artifacts' / 'baseline' / 'lightgbm_model.joblib')
    tfidf_vectorizer = joblib.load(project_root / 'training' / 'artifacts' / 'baseline' / 'tfidf_vectorizer.joblib')
    X_tfidf = tfidf_vectorizer.transform(test_df['review_text'].fillna(''))
    lgb_base_probs = lgb_base_model.predict_proba(X_tfidf)[:, 1]

    # 5. Hybrid Deep Models (DeBERTaTabularFusionNet)
    hybrid_dir = project_root / 'backend' / 'models' / 'hybrid_deberta_tabular'
    scaler_bundle = joblib.load(hybrid_dir / 'tabular_scaler.joblib')
    scaler = scaler_bundle['scaler']
    X_tab_scaled, feat_cols = extract_tabular_features(test_df, scaler=scaler)
    test_emb = np.load(project_root / 'training' / 'artifacts' / 'cache' / 'test_embeddings.npy')

    device = torch.device('cpu')
    ckpt_cw = torch.load(hybrid_dir / 'pytorch_hybrid_class_weighted.pt', map_location=device, weights_only=False)
    model_cw = DeBERTaTabularFusionNet(
        text_dim=ckpt_cw['text_dim'],
        tab_dim=ckpt_cw['tab_dim'],
        hidden_dim=ckpt_cw['hidden_dim'],
    )
    model_cw.load_state_dict(ckpt_cw['model_state_dict'])
    model_cw.eval()

    ckpt_focal = torch.load(hybrid_dir / 'pytorch_hybrid_focal_loss.pt', map_location=device, weights_only=False)
    model_focal = DeBERTaTabularFusionNet(
        text_dim=ckpt_focal['text_dim'],
        tab_dim=ckpt_focal['tab_dim'],
        hidden_dim=ckpt_focal['hidden_dim'],
    )
    model_focal.load_state_dict(ckpt_focal['model_state_dict'])
    model_focal.eval()

    with torch.no_grad():
        t_emb = torch.tensor(test_emb, dtype=torch.float32)
        t_tab = torch.tensor(X_tab_scaled, dtype=torch.float32)
        logits_cw = model_cw(t_emb, t_tab)
        probs_cw = torch.softmax(logits_cw, dim=-1)[:, 1].numpy()
        logits_focal = model_focal(t_emb, t_tab)
        probs_focal = torch.softmax(logits_focal, dim=-1)[:, 1].numpy()

    # 6. LightGBM Multimodal Hybrid
    lgb_hybrid_model = joblib.load(hybrid_dir / 'lgb_hybrid_standard.joblib')
    X_hybrid = np.column_stack([deb_probs, X_tab_scaled])
    probs_lgb_hybrid = lgb_hybrid_model.predict_proba(X_hybrid)[:, 1]

    models_probs = {
        'TF-IDF + LightGBM Baseline': lgb_base_probs,
        'Calibrated Linear SVM': svm_probs,
        'Packed BiLSTM': bilstm_probs,
        'DeBERTa-v3-base (Text-Only)': deb_probs,
        'LightGBM Multimodal Hybrid': probs_lgb_hybrid,
        'DeBERTaTabularFusionNet (Class-Weighted)': probs_cw,
        'DeBERTaTabularFusionNet (Focal Loss)': probs_focal,
    }

    return test_df, y_true, models_probs, feat_cols, lgb_base_model, tfidf_vectorizer, lgb_hybrid_model


def save_figure(fig, out_dir: Path, basename: str):
    """Save both high-res PNG and vector PDF."""
    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {png_path.name} and {pdf_path.name}")


def generate_fig1_performance_comparison(y_true, models_probs, out_dir: Path):
    """Figure 1: Grouped bar chart comparing key evaluation metrics across all verified models."""
    records = []
    for name, probs in models_probs.items():
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        roc = roc_auc_score(y_true, probs)
        pr = average_precision_score(y_true, probs)
        records.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc,
            'PR-AUC': pr,
        })

    df_metrics = pd.DataFrame(records)
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(df_metrics))
    width = 0.13

    palette = ['#2b5c8f', '#4682b4', '#d95f02', '#e7298a', '#1b9e77', '#7570b3']
    for i, col in enumerate(metric_cols):
        offset = (i - len(metric_cols) / 2 + 0.5) * width
        rects = ax.bar(x + offset, df_metrics[col], width, label=col, color=palette[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score [0.0 - 1.0]')
    ax.set_title('Comprehensive Model Benchmark on Held-Out Test Set (4,474 Reviews, Threshold = 0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['Model'], rotation=18, ha='right', fontsize=9.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='upper left', ncol=6, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    save_figure(fig, out_dir, "fig1_model_performance_comparison")


def generate_fig2_combined_roc_curves(y_true, models_probs, out_dir: Path):
    """Figure 2: Publication-quality Combined ROC curves on identical 4,474 test reviews."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        color = MODEL_COLORS.get(name, '#333333')
        lw = 2.2 if 'Fusion' in name or 'DeBERTa' in name else 1.6
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=lw)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.7, label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves\n(Identical Held-Out Test Partition: N = 4,474)')
    ax.legend(loc='lower right', frameon=True, framealpha=0.92, facecolor='white')
    ax.grid(True, linestyle='--', alpha=0.6)

    save_figure(fig, out_dir, "fig2_combined_roc_curves")


def generate_fig3_combined_pr_curves(y_true, models_probs, out_dir: Path):
    """Figure 3: Combined Precision-Recall Curves with empirical class prevalence baseline."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    prevalence = y_true.sum() / len(y_true)

    for name, probs in models_probs.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        color = MODEL_COLORS.get(name, '#333333')
        lw = 2.2 if 'Fusion' in name or 'DeBERTa' in name else 1.6
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=lw)

    ax.axhline(prevalence, color='k', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Class Prevalence Baseline ({prevalence:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall (Sensitivity for Deceptive Reviews)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall (PR) Curves for Fake Review Detection\n(Held-Out Test Partition: 1,677 Fake / 2,797 Genuine)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, facecolor='white')
    ax.grid(True, linestyle='--', alpha=0.6)

    save_figure(fig, out_dir, "fig3_combined_precision_recall_curves")


def generate_fig4_confusion_matrices(y_true, models_probs, out_dir: Path):
    """Figure 4: 2x3 Grid of Normalized and Raw Confusion Matrices."""
    selected_models = [
        ('DeBERTa-v3 Text (Default, τ=0.50)', models_probs['DeBERTa-v3-base (Text-Only)'], 0.50),
        ('DeBERTa-v3 Text (Val-Tuned, τ=0.12)', models_probs['DeBERTa-v3-base (Text-Only)'], 0.12),
        ('TF-IDF + LightGBM (Default, τ=0.50)', models_probs['TF-IDF + LightGBM Baseline'], 0.50),
        ('Packed BiLSTM (Default, τ=0.50)', models_probs['Packed BiLSTM'], 0.50),
        ('Linear SVM (Default, τ=0.50)', models_probs['Calibrated Linear SVM'], 0.50),
        ('FusionNet Focal Loss (Default, τ=0.50)', models_probs['DeBERTaTabularFusionNet (Focal Loss)'], 0.50),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, (title, probs, thresh) in enumerate(selected_models):
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            annot_kws={'size': 13, 'weight': 'bold'},
        )

        # Annotate percentages inside heatmap cells
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5,
                    i + 0.72,
                    f"({cm_norm[i, j]*100:.1f}%)",
                    ha='center',
                    va='center',
                    color='darkred' if cm_norm[i, j] < 0.4 else 'white',
                    fontsize=9.5,
                )

        ax.set_title(title, pad=10, fontsize=11)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_xticklabels(['Genuine (0)', 'Fake (1)'])
        ax.set_yticklabels(['Genuine (0)', 'Fake (1)'], rotation=0)

    fig.suptitle('Test Confusion Matrices (Actual Counts and Class-Normalized Percentages)', y=0.99, fontsize=15)
    plt.tight_layout()
    save_figure(fig, out_dir, "fig4_confusion_matrices")


def generate_fig5_threshold_sensitivity(y_true, models_probs, out_dir: Path):
    """Figure 5: Threshold Sensitivity Curves (Threshold vs Precision, Recall, F1)."""
    thresholds = np.linspace(0.02, 0.98, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Subplot A: DeBERTa-v3 Text-Only
    deb_probs = models_probs['DeBERTa-v3-base (Text-Only)']
    deb_p, deb_r, deb_f1 = [], [], []
    for t in thresholds:
        p = (deb_probs >= t).astype(int)
        deb_p.append(precision_score(y_true, p, zero_division=0))
        deb_r.append(recall_score(y_true, p, zero_division=0))
        deb_f1.append(f1_score(y_true, p, zero_division=0))

    ax0 = axes[0]
    ax0.plot(thresholds, deb_p, label='Precision', color='#2b5c8f', linewidth=2.0)
    ax0.plot(thresholds, deb_r, label='Recall', color='#d95f02', linewidth=2.0)
    ax0.plot(thresholds, deb_f1, label='F1-Score', color='#1b9e77', linewidth=2.4)
    ax0.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (0.50)')
    ax0.axvline(0.12, color='crimson', linestyle='--', label='Val-Tuned Optimal (0.12)')
    ax0.set_xlabel('Decision Threshold')
    ax0.set_ylabel('Metric Value')
    ax0.set_title('DeBERTa-v3-base (Text-Only)\nThreshold Sensitivity Analysis')
    ax0.legend(loc='center right', frameon=True)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # Subplot B: DeBERTaTabularFusionNet Focal Loss
    fusion_probs = models_probs['DeBERTaTabularFusionNet (Focal Loss)']
    fus_p, fus_r, fus_f1 = [], [], []
    for t in thresholds:
        p = (fusion_probs >= t).astype(int)
        fus_p.append(precision_score(y_true, p, zero_division=0))
        fus_r.append(recall_score(y_true, p, zero_division=0))
        fus_f1.append(f1_score(y_true, p, zero_division=0))

    ax1 = axes[1]
    ax1.plot(thresholds, fus_p, label='Precision', color='#2b5c8f', linewidth=2.0)
    ax1.plot(thresholds, fus_r, label='Recall', color='#d95f02', linewidth=2.0)
    ax1.plot(thresholds, fus_f1, label='F1-Score', color='#1b9e77', linewidth=2.4)
    ax1.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (0.50)')
    ax1.axvline(0.41, color='crimson', linestyle='--', label='Val-Tuned Optimal (0.41)')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_title('DeBERTaTabularFusionNet (Focal Loss)\nThreshold Sensitivity Analysis')
    ax1.legend(loc='center right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, out_dir, "fig5_threshold_sensitivity_analysis")


def generate_fig6_correlation_heatmap(test_df: pd.DataFrame, out_dir: Path):
    """Figure 6: Genuine Pearson Correlation Heatmap of real dataset metadata features."""
    # Compute genuine numerical metadata features
    df_feat, feat_cols = extract_tabular_features(test_df)
    df_feat_pd = pd.DataFrame(df_feat, columns=feat_cols)

    corr = df_feat_pd.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='vlag',
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.6,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation Coefficient'},
        ax=ax,
        annot_kws={'size': 8.5},
    )

    ax.set_title('Empirical Correlation Matrix of Engineered Behavioral & Metadata Features\n(Computed on Genuine 4,474 Test Split Reviews)', pad=16)
    plt.xticks(rotation=40, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    save_figure(fig, out_dir, "fig6_behavioral_correlation_heatmap")


def generate_fig7_feature_importance(lgb_base_model, tfidf_vectorizer, lgb_hybrid_model, feat_cols, out_dir: Path):
    """Figure 7: Genuine Feature Importance from trained LightGBM models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Subplot A: Baseline Top 15 TF-IDF Terms
    terms = np.array(tfidf_vectorizer.get_feature_names_out())
    importances_base = lgb_base_model.feature_importances_
    top_indices = np.argsort(importances_base)[-15:]

    top_terms = terms[top_indices]
    top_scores = importances_base[top_indices]

    ax0 = axes[0]
    ax0.barh(top_terms, top_scores, color='#4c72b0', edgecolor='black', linewidth=0.6)
    ax0.set_xlabel('Split Importance (Splits Count in LightGBM Trees)')
    ax0.set_title('Top 15 Predictive Terms in Baseline Model\n(TF-IDF + LightGBM)')
    ax0.grid(axis='x', linestyle='--', alpha=0.5)

    # Subplot B: Multimodal Hybrid Feature Importances
    hybrid_feature_names = ['deberta_text_probability'] + feat_cols
    importances_hybrid = lgb_hybrid_model.feature_importances_
    sorted_hybrid_idx = np.argsort(importances_hybrid)

    ax1 = axes[1]
    bar_colors = ['#c44e52' if name == 'deberta_text_probability' else '#8172b3'
                  for name in np.array(hybrid_feature_names)[sorted_hybrid_idx]]
    ax1.barh(np.array(hybrid_feature_names)[sorted_hybrid_idx],
             importances_hybrid[sorted_hybrid_idx],
             color=bar_colors, edgecolor='black', linewidth=0.6)
    ax1.set_xlabel('Split Importance')
    ax1.set_title('Feature Importance in Multimodal Hybrid Model\n(DeBERTa Text Probability + Tabular Features)')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c44e52', edgecolor='black', label='DeBERTa Contextual Representation'),
        Patch(facecolor='#8172b3', edgecolor='black', label='Tabular Behavioral Feature')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True)

    plt.tight_layout()
    save_figure(fig, out_dir, "fig7_feature_importance_lightgbm")


def generate_fig8_standardized_differences(test_df: pd.DataFrame, feat_cols: list, out_dir: Path):
    """Figure 8: Empirical Standardized Effect Sizes (Cohen's d) for Behavioral Features."""
    df_feat, _ = extract_tabular_features(test_df)
    df_feat_pd = pd.DataFrame(df_feat, columns=feat_cols)
    df_feat_pd['is_fake'] = (test_df['label'].astype(str).str.lower() == 'fake').astype(int).values

    diffs = []
    for col in feat_cols:
        gen_vals = df_feat_pd[df_feat_pd['is_fake'] == 0][col].values
        fake_vals = df_feat_pd[df_feat_pd['is_fake'] == 1][col].values

        mean_gen = np.mean(gen_vals)
        mean_fake = np.mean(fake_vals)
        pooled_std = np.sqrt((np.var(gen_vals, ddof=1) + np.var(fake_vals, ddof=1)) / 2.0) + 1e-9
        cohen_d = (mean_fake - mean_gen) / pooled_std

        diffs.append({
            'feature': col,
            'cohen_d': cohen_d,
            'abs_d': abs(cohen_d),
        })

    df_d = pd.DataFrame(diffs).sort_values(by='abs_d', ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bar_colors = ['#d95f02' if d > 0 else '#2b5c8f' for d in df_d['cohen_d']]

    ax.barh(df_d['feature'], df_d['cohen_d'], color=bar_colors, edgecolor='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=1.0)
    ax.set_xlabel("Cohen's d: Standardized Mean Difference [(Fake - Genuine) / Pooled Std]")
    ax.set_title("Empirical Behavioral Discrepancies between Genuine and Deceptive Reviews\n(Computed on 4,474 Real E-Commerce Reviews)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d95f02', edgecolor='black', label='Elevated in Deceptive Reviews'),
        Patch(facecolor='#2b5c8f', edgecolor='black', label='Elevated in Authentic Reviews')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)

    plt.tight_layout()
    save_figure(fig, out_dir, "fig8_behavioral_standardized_differences")


def main():
    out_dir = PROJECT_ROOT / 'results' / 'figures' / 'research_paper'
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Generating Publication-Quality Figures from Genuine Benchmark Data")
    print(f"Output directory: {out_dir}")
    print("=" * 70)

    test_df, y_true, models_probs, feat_cols, lgb_base_model, tfidf_vec, lgb_hybrid_model = load_test_data_and_predictions(PROJECT_ROOT)

    print("\n1. Generating Figure 1: Model Performance Comparison Bar Chart...")
    generate_fig1_performance_comparison(y_true, models_probs, out_dir)

    print("\n2. Generating Figure 2: Combined ROC Curves...")
    generate_fig2_combined_roc_curves(y_true, models_probs, out_dir)

    print("\n3. Generating Figure 3: Combined Precision-Recall Curves...")
    generate_fig3_combined_pr_curves(y_true, models_probs, out_dir)

    print("\n4. Generating Figure 4: Multi-Model Confusion Matrices...")
    generate_fig4_confusion_matrices(y_true, models_probs, out_dir)

    print("\n5. Generating Figure 5: Decision Threshold Sensitivity Analysis...")
    generate_fig5_threshold_sensitivity(y_true, models_probs, out_dir)

    print("\n6. Generating Figure 6: Behavioral Feature Correlation Matrix...")
    generate_fig6_correlation_heatmap(test_df, out_dir)

    print("\n7. Generating Figure 7: LightGBM Feature Importance (Baseline & Hybrid)...")
    generate_fig7_feature_importance(lgb_base_model, tfidf_vec, lgb_hybrid_model, feat_cols, out_dir)

    print("\n8. Generating Figure 8: Standardized Behavioral Effect Sizes (Cohen's d)...")
    generate_fig8_standardized_differences(test_df, feat_cols, out_dir)

    print("\n" + "=" * 70)
    print(f"All 8 research publication figures successfully generated in {out_dir} (PNG & PDF)")
    print("=" * 70)


if __name__ == '__main__':
    main()
