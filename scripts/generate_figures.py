"""Generate genuine, publication-quality figures and tables for the Fake Review Detection research paper.
Strictly uses empirical predictions on identical held-out test splits without simulation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.generate_publication_figures import (
    load_test_data_and_predictions,
    generate_fig1_performance_comparison,
    generate_fig2_combined_roc_curves,
    generate_fig3_combined_pr_curves,
    generate_fig4_confusion_matrices,
    generate_fig5_threshold_sensitivity,
    generate_fig6_correlation_heatmap,
    generate_fig7_feature_importance,
    generate_fig8_standardized_differences,
)


def generate_paper_figures(output_dir: str = "results/figures/research_paper"):
    out_path = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating genuine research paper figures from verified test data in {out_path}...")
    test_df, y_true, models_probs, feat_cols, lgb_base, tfidf_vec, lgb_hybrid = load_test_data_and_predictions(project_root)

    generate_fig1_performance_comparison(y_true, models_probs, out_path)
    generate_fig2_combined_roc_curves(y_true, models_probs, out_path)
    generate_fig3_combined_pr_curves(y_true, models_probs, out_path)
    generate_fig4_confusion_matrices(y_true, models_probs, out_path)
    generate_fig5_threshold_sensitivity(y_true, models_probs, out_path)
    generate_fig6_correlation_heatmap(test_df, out_path)
    generate_fig7_feature_importance(lgb_base, tfidf_vec, lgb_hybrid, feat_cols, out_path)
    generate_fig8_standardized_differences(test_df, feat_cols, out_path)
    print(f"Completed! All genuine figures generated in {out_path}")


if __name__ == "__main__":
    generate_paper_figures()

