"""
metrics.py
Comprehensive metrics module for the FakeGuard fake review detection project.
Calculates, stores, and plots evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class MetricsCalculator:
    """Calculates and stores evaluation metrics for fake review detection."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None):
        """
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities for the positive class.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all standard metrics."""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro'),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro'),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro'),
            'f1_binary': f1_score(self.y_true, self.y_pred, average='binary', zero_division=0)
        }
        
        if self.y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_prob)
            metrics['auc_pr'] = average_precision_score(self.y_true, self.y_prob)
            
        return metrics
    
    def get_classification_report(self) -> str:
        """Get sklearn classification report."""
        return classification_report(self.y_true, self.y_pred)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, title: str = 'Confusion Matrix'):
        """Plot and optionally save confusion matrix heatmap."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Genuine', 'Fake'], 
                    yticklabels=['Genuine', 'Fake'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve with AUC score."""
        if self.y_prob is None:
            raise ValueError("Probabilities must be provided to plot ROC curve.")
            
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        auc_score = roc_auc_score(self.y_true, self.y_prob)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'Model (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None):
        """Plot precision-recall curve."""
        if self.y_prob is None:
            raise ValueError("Probabilities must be provided to plot PR curve.")
            
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        ap_score = average_precision_score(self.y_true, self.y_prob)
        
        plt.figure()
        plt.plot(recall, precision, label=f'Model (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class ModelComparator:
    """Compares multiple models' performance for paper tables."""
    
    def __init__(self):
        self.results = {}
        self.rocs = {}
    
    def add_model(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None):
        """Add a model's results."""
        calc = MetricsCalculator(y_true, y_pred, y_prob)
        self.results[name] = calc.compute_all()
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            self.rocs[name] = (fpr, tpr, auc)
    
    def comparison_table(self) -> pd.DataFrame:
        """Generate comparison table (like Table 1 in papers)."""
        df = pd.DataFrame.from_dict(self.results, orient='index')
        return df.round(4)
    
    def plot_comparison_bar(self, metrics: List[str] = ['accuracy', 'f1_macro', 'auc_roc'], save_path: Optional[str] = None):
        """Bar chart comparing models across metrics."""
        df = self.comparison_table()
        
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            raise ValueError("None of the specified metrics are available.")
            
        df_filtered = df[available_metrics]
        
        ax = df_filtered.plot(kind='bar', figsize=(10, 6), rot=45)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(title='Metrics')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.1)
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_all_roc_curves(self, save_path: Optional[str] = None):
        """Overlay ROC curves for all models on one plot."""
        if not self.rocs:
            raise ValueError("No ROC data available. Probabilities must be provided when adding models.")
            
        plt.figure(figsize=(8, 8))
        
        for name, (fpr, tpr, auc) in self.rocs.items():
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def to_latex(self, save_path: Optional[str] = None) -> str:
        """Export comparison table as LaTeX for the paper."""
        df = self.comparison_table()
        latex_str = df.to_latex(float_format="%.4f")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex_str)
                
        return latex_str
