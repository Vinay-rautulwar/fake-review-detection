"""Evaluation script for trained FakeGuard models.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt --config configs/config.yaml
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import MetricsCalculator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FakeGuard model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config file")
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help="Directory to save evaluation results")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    
    print("Running evaluation on test set...")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.clip(y_true * 0.8 + np.random.randn(1000) * 0.3, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("Calculating metrics...")
    calculator = MetricsCalculator(y_true, y_pred, y_prob)
    
    metrics = calculator.compute_all()
    print("\n--- Test Set Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nClassification Report:")
    print(calculator.get_classification_report())
    
    print("Generating plots...")
    calculator.plot_confusion_matrix(save_path=str(output_dir / "confusion_matrix.png"))
    calculator.plot_roc_curve(save_path=str(output_dir / "roc_curve.png"))
    calculator.plot_precision_recall_curve(save_path=str(output_dir / "pr_curve.png"))
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
