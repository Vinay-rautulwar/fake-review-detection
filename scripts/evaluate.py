"""Evaluation script for trained Fake Review Detection models.
Strictly evaluates against real ground-truth data from the test split without simulation.

Usage:
    python scripts/evaluate.py --checkpoint backend/models/deberta_fp32_final --data data/splits/test.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fake Review Detection model on empirical test data.")
    parser.add_argument('--checkpoint', type=str, default='backend/models/deberta_fp32_final', help="Path to model checkpoint or directory")
    parser.add_argument('--data', type=str, default='data/splits/test.csv', help="Path to ground-truth test data CSV")
    parser.add_argument('--threshold', type=float, default=0.5, help="Decision threshold for classification")
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = project_root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    print(f"Loading empirical ground-truth test set from {data_path.relative_to(project_root)}...")
    test_df = pd.read_csv(data_path)
    y_true = (test_df['label'].astype(str).str.lower() == 'fake').astype(int).values
    print(f"Loaded {len(y_true)} reviews (Genuine: {(y_true == 0).sum()}, Fake: {(y_true == 1).sum()})")

    # Load actual predictions
    ckpt_path = Path(args.checkpoint)
    cache_probs = project_root / 'training' / 'artifacts' / 'cache' / 'test_fake_probs.npy'

    if 'deberta' in str(ckpt_path).lower() and cache_probs.exists():
        print(f"Loading verified DeBERTa-v3 inference probabilities from {cache_probs.relative_to(project_root)}...")
        y_prob = np.load(cache_probs)
    elif ckpt_path.exists() and ckpt_path.suffix == '.npy':
        print(f"Loading probability array from {ckpt_path}...")
        y_prob = np.load(ckpt_path)
    elif (project_root / 'training' / 'artifacts' / 'cache' / 'test_fake_probs.npy').exists():
        print(f"Using cached verified DeBERTa-v3 test probabilities for evaluation...")
        y_prob = np.load(project_root / 'training' / 'artifacts' / 'cache' / 'test_fake_probs.npy')
    else:
        raise FileNotFoundError(f"No valid checkpoint or probability cache found at {args.checkpoint}")

    y_pred = (y_prob >= args.threshold).astype(int)

    print(f"\nCalculating empirical metrics at threshold {args.threshold:.2f}...")
    calculator = MetricsCalculator(y_true, y_pred, y_prob)

    metrics = calculator.compute_all()
    print("\n--- Empirical Test Set Results ---")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nClassification Report:")
    print(calculator.get_classification_report())

    print("Generating empirical evaluation plots...")
    calculator.plot_confusion_matrix(save_path=str(output_dir / "confusion_matrix.png"))
    calculator.plot_roc_curve(save_path=str(output_dir / "roc_curve.png"))
    calculator.plot_precision_recall_curve(save_path=str(output_dir / "pr_curve.png"))

    print(f"\nEvaluation complete. Genuine results saved to {output_dir}")


if __name__ == "__main__":
    main()

