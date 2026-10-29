"""
Fake Review Detection - TF-IDF + LightGBM Baseline Training Script
===================================================================
Step 5: Train and Evaluate the Text-Only Baseline Model.

This script trains a TF-IDF (word unigrams + bigrams) + LightGBM binary classifier
using strictly review_text on reviewer-held-out train.csv and evaluates on validation.csv.

Constraints:
1. Does NOT load or touch test.csv (reserved for final evaluation).
2. Does NOT use reviewer_id, product_id, or aggregate metadata columns.
3. Normalizes target: genuine = 0, fake = 1.
4. Evaluates at threshold 0.5 and optimal validation-selected threshold.
5. Saves model artifacts under training/artifacts/baseline/ and report under reports/baseline_validation_report.txt.

Usage:
    python train_baseline.py
"""

import argparse
from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import yaml


class BaselineTrainer:
    """Trains and evaluates the TF-IDF + LightGBM text baseline model."""

    def __init__(
        self,
        train_path: str = "data/splits/train.csv",
        val_path: str = "data/splits/validation.csv",
        artifacts_dir: str = "training/artifacts/baseline",
        reports_dir: str = "reports",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 3,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 300,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: str = "balanced",
        seed: int = 42,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.artifacts_dir = artifacts_dir
        self.reports_dir = reports_dir

        # TF-IDF hyperparameters
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # LightGBM hyperparameters
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.seed = seed

        # State
        self.vectorizer: TfidfVectorizer = None
        self.model: lgb.LGBMClassifier = None
        self.results: Dict[str, Any] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and validation splits, ensuring test split is NOT accessed."""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training split not found at: {self.train_path}")
        if not os.path.exists(self.val_path):
            raise FileNotFoundError(f"Validation split not found at: {self.val_path}")

        print(f"[1/5] Loading datasets:")
        print(f"      Train path      : {self.train_path}")
        print(f"      Validation path : {self.val_path}")

        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)

        # Enforce that only review_text and label are used for training
        assert "review_text" in df_train.columns and "label" in df_train.columns
        assert "review_text" in df_val.columns and "label" in df_val.columns

        print(f"      Train shape     : {df_train.shape[0]:,} rows x {df_train.shape[1]} cols")
        print(f"      Validation shape: {df_val.shape[0]:,} rows x {df_val.shape[1]} cols")
        print("      [CHECK] test.csv was NOT loaded (reserved for final evaluation).")

        return df_train, df_val

    def extract_features(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame
    ) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        """Fit TF-IDF vectorizer strictly on train text and transform validation text."""
        print(f"[2/5] Fitting TF-IDF Vectorizer on training text...")
        print(f"      N-gram range   : {self.ngram_range}")
        print(f"      Max features   : {self.max_features:,}")
        print(f"      Sublinear TF   : {self.sublinear_tf}")

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )

        X_train_tfidf = self.vectorizer.fit_transform(df_train["review_text"])
        X_val_tfidf = self.vectorizer.transform(df_val["review_text"])

        # Map target labels: genuine = 0, fake = 1
        label_map = {"genuine": 0, "fake": 1}
        y_train = df_train["label"].map(label_map).values
        y_val = df_val["label"].map(label_map).values

        actual_vocab_size = len(self.vectorizer.vocabulary_)
        print(f"      Vocabulary size: {actual_vocab_size:,} features")
        print(f"      Train matrix   : {X_train_tfidf.shape[0]:,} x {X_train_tfidf.shape[1]:,} (sparse CSR)")
        print(f"      Val matrix     : {X_val_tfidf.shape[0]:,} x {X_val_tfidf.shape[1]:,} (sparse CSR)")

        return X_train_tfidf, X_val_tfidf, y_train, y_val

    def train_model(self, X_train_tfidf, y_train) -> None:
        """Train LightGBM binary classifier using training set sparse matrix."""
        print(f"[3/5] Training LightGBM binary classifier (seed={self.seed})...")
        print(f"      Learning rate  : {self.learning_rate}")
        print(f"      Num leaves     : {self.num_leaves}")
        print(f"      N estimators   : {self.n_estimators}")
        print(f"      Class weight   : {self.class_weight}")

        self.model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )

        self.model.fit(X_train_tfidf, y_train)
        print("      [OK] Model training completed.")

    def evaluate_validation(self, X_val_tfidf, y_val) -> Dict[str, Any]:
        """Evaluate validation set at default 0.50 threshold and optimal threshold."""
        print("[4/5] Evaluating model performance on validation set...")
        y_val_prob = self.model.predict_proba(X_val_tfidf)[:, 1]

        roc_auc = float(roc_auc_score(y_val, y_val_prob))
        pr_auc = float(average_precision_score(y_val, y_val_prob))

        # 1. Default threshold = 0.50
        y_pred_05 = (y_val_prob >= 0.50).astype(int)
        cm_05 = confusion_matrix(y_val, y_pred_05).tolist()
        p_05 = float(precision_score(y_val, y_pred_05, zero_division=0))
        r_05 = float(recall_score(y_val, y_pred_05, zero_division=0))
        f1_05 = float(f1_score(y_val, y_pred_05, zero_division=0))
        clf_report_05 = classification_report(
            y_val, y_pred_05, target_names=["genuine (0)", "fake (1)"], output_dict=True
        )

        # 2. Optimal threshold search on validation set (maximizing F1)
        best_threshold = 0.50
        best_f1 = f1_05
        threshold_trials = []
        for t in np.arange(0.10, 0.91, 0.01):
            t_rounded = round(float(t), 2)
            y_pred_t = (y_val_prob >= t_rounded).astype(int)
            f1_t = float(f1_score(y_val, y_pred_t, zero_division=0))
            p_t = float(precision_score(y_val, y_pred_t, zero_division=0))
            r_t = float(recall_score(y_val, y_pred_t, zero_division=0))
            threshold_trials.append({"threshold": t_rounded, "f1": f1_t, "precision": p_t, "recall": r_t})
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t_rounded

        y_pred_best = (y_val_prob >= best_threshold).astype(int)
        cm_best = confusion_matrix(y_val, y_pred_best).tolist()
        p_best = float(precision_score(y_val, y_pred_best, zero_division=0))
        r_best = float(recall_score(y_val, y_pred_best, zero_division=0))
        clf_report_best = classification_report(
            y_val, y_pred_best, target_names=["genuine (0)", "fake (1)"], output_dict=True
        )

        self.results = {
            "model_type": "TF-IDF + LightGBM",
            "feature_scope": "review_text (text-only)",
            "evaluation_split": "validation.csv",
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold_0_5": {
                "threshold": 0.50,
                "precision": round(p_05, 4),
                "recall": round(r_05, 4),
                "f1_score": round(f1_05, 4),
                "confusion_matrix": {
                    "tn": cm_05[0][0],
                    "fp": cm_05[0][1],
                    "fn": cm_05[1][0],
                    "tp": cm_05[1][1],
                },
                "classification_report": clf_report_05,
            },
            "optimal_threshold": {
                "threshold": best_threshold,
                "precision": round(p_best, 4),
                "recall": round(r_best, 4),
                "f1_score": round(best_f1, 4),
                "confusion_matrix": {
                    "tn": cm_best[0][0],
                    "fp": cm_best[0][1],
                    "fn": cm_best[1][0],
                    "tp": cm_best[1][1],
                },
                "classification_report": clf_report_best,
            },
        }

        print("\n--- Validation Performance Summary ---")
        print(f"ROC-AUC           : {roc_auc:.4f}")
        print(f"PR-AUC            : {pr_auc:.4f}")
        print(f"[Threshold 0.50]  : Precision={p_05:.4f}, Recall={r_05:.4f}, F1={f1_05:.4f}")
        print(f"[Optimal {best_threshold:.2f}] : Precision={p_best:.4f}, Recall={r_best:.4f}, F1={best_f1:.4f}")

        return self.results

    def save_artifacts_and_report(self) -> None:
        """Export trained model artifacts and write validation report."""
        print("[5/5] Saving model artifacts and reports...")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # 1. Save Vectorizer
        vec_path = os.path.join(self.artifacts_dir, "tfidf_vectorizer.joblib")
        joblib.dump(self.vectorizer, vec_path)
        print(f"      [OK] Saved vectorizer      : {vec_path}")

        # 2. Save LightGBM Model
        model_path = os.path.join(self.artifacts_dir, "lightgbm_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"      [OK] Saved LightGBM model  : {model_path}")

        # 3. Save Configuration
        config_data = {
            "model_type": "tfidf_lightgbm",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "data": {
                "train_path": self.train_path,
                "val_path": self.val_path,
                "test_path": "RESERVED_NOT_LOADED",
                "input_feature": "review_text",
                "target_feature": "label (genuine=0, fake=1)",
            },
            "tfidf": {
                "ngram_range": list(self.ngram_range),
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "sublinear_tf": self.sublinear_tf,
            },
            "lightgbm": {
                "objective": "binary",
                "boosting_type": "gbdt",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "n_estimators": self.n_estimators,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "class_weight": self.class_weight,
                "random_state": self.seed,
            },
        }
        config_path = os.path.join(self.artifacts_dir, "model_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        print(f"      [OK] Saved model config    : {config_path}")

        # 4. Save Metrics JSON
        metrics_path = os.path.join(self.artifacts_dir, "validation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"      [OK] Saved metrics JSON    : {metrics_path}")

        # 5. Write Validation Report TXT
        report_path = os.path.join(self.reports_dir, "baseline_validation_report.txt")
        self._write_report_file(report_path)
        print(f"      [OK] Saved report text     : {report_path}")

    def _write_report_file(self, report_path: str) -> None:
        """Write human-readable validation report."""
        res = self.results
        m05 = res["threshold_0_5"]
        mb = res["optimal_threshold"]
        cm05 = m05["confusion_matrix"]
        cmb = mb["confusion_matrix"]

        lines = [
            "=" * 78,
            " FAKE REVIEW DETECTION - BASELINE MODEL VALIDATION REPORT (STEP 5)",
            "=" * 78,
            f"Execution Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model Architecture    : TF-IDF (word unigrams + bigrams) + LightGBM (GBDT)",
            f"Predictive Input      : review_text (Text-Only Baseline)",
            f"Target Mapping        : genuine = 0, fake = 1",
            f"Training Data Split   : {self.train_path} (20,875 reviews)",
            f"Validation Data Split : {self.val_path} (4,472 reviews)",
            f"Test Split Status     : RESERVED (data/splits/test.csv was NOT loaded or evaluated)",
            f"Random Seed           : {self.seed}",
            "-" * 78,
            "1. HYPERPARAMETER CONFIGURATION",
            "-" * 78,
            "TF-IDF Feature Extraction:",
            f"  - N-gram Range       : {self.ngram_range}",
            f"  - Max Features       : {self.max_features:,}",
            f"  - Min / Max DF       : min_df={self.min_df}, max_df={self.max_df}",
            f"  - Sublinear TF       : {self.sublinear_tf} (1 + log(tf))",
            "",
            "LightGBM Classifier:",
            f"  - Objective          : binary",
            f"  - Boosting Type      : gbdt",
            f"  - Learning Rate      : {self.learning_rate}",
            f"  - Number of Leaves   : {self.num_leaves}",
            f"  - N Estimators       : {self.n_estimators}",
            f"  - Subsample (bagging): {self.subsample}",
            f"  - Colsample by Tree  : {self.colsample_bytree}",
            f"  - Class Weight       : {self.class_weight}",
            f"  - Random State       : {self.seed}",
            "-" * 78,
            "2. VALIDATION PERFORMANCE OVERVIEW",
            "-" * 78,
            f"ROC-AUC Score          : {res['roc_auc']:.4f}",
            f"PR-AUC (Avg Precision) : {res['pr_auc']:.4f}",
            "",
            "Comparison of Classification Thresholds:",
            f"{'Metric':<25} | {'Default (0.50)':<18} | {'Validation-Optimal (' + str(mb['threshold']) + ')':<25}",
            "-" * 78,
            f"{'Precision (Fake)':<25} | {m05['precision']:<18.4f} | {mb['precision']:<25.4f}",
            f"{'Recall (Fake)':<25} | {m05['recall']:<18.4f} | {mb['recall']:<25.4f}",
            f"{'F1-Score (Fake)':<25} | {m05['f1_score']:<18.4f} | {mb['f1_score']:<25.4f}",
            "-" * 78,
            "3. CONFUSION MATRICES",
            "-" * 78,
            f"A. At Default Threshold (0.50):",
            f"   True Negative (Genuine as Genuine) : {cm05['tn']:>5}  |  False Positive (Genuine as Fake) : {cm05['fp']:>5}",
            f"   False Negative (Fake as Genuine)   : {cm05['fn']:>5}  |  True Positive (Fake as Fake)     : {cm05['tp']:>5}",
            "",
            f"B. At Optimal Threshold ({mb['threshold']}):",
            f"   True Negative (Genuine as Genuine) : {cmb['tn']:>5}  |  False Positive (Genuine as Fake) : {cmb['fp']:>5}",
            f"   False Negative (Fake as Genuine)   : {cmb['fn']:>5}  |  True Positive (Fake as Fake)     : {cmb['tp']:>5}",
            "-" * 78,
            "4. CLASSIFICATION REPORTS",
            "-" * 78,
            "Default Threshold (0.50):",
            f"  - Genuine (0) : Precision = {m05['classification_report']['genuine (0)']['precision']:.4f}, Recall = {m05['classification_report']['genuine (0)']['recall']:.4f}, F1 = {m05['classification_report']['genuine (0)']['f1-score']:.4f}",
            f"  - Fake (1)    : Precision = {m05['classification_report']['fake (1)']['precision']:.4f}, Recall = {m05['classification_report']['fake (1)']['recall']:.4f}, F1 = {m05['classification_report']['fake (1)']['f1-score']:.4f}",
            f"  - Macro Avg   : Precision = {m05['classification_report']['macro avg']['precision']:.4f}, Recall = {m05['classification_report']['macro avg']['recall']:.4f}, F1 = {m05['classification_report']['macro avg']['f1-score']:.4f}",
            "",
            f"Optimal Threshold ({mb['threshold']}):",
            f"  - Genuine (0) : Precision = {mb['classification_report']['genuine (0)']['precision']:.4f}, Recall = {mb['classification_report']['genuine (0)']['recall']:.4f}, F1 = {mb['classification_report']['genuine (0)']['f1-score']:.4f}",
            f"  - Fake (1)    : Precision = {mb['classification_report']['fake (1)']['precision']:.4f}, Recall = {mb['classification_report']['fake (1)']['recall']:.4f}, F1 = {mb['classification_report']['fake (1)']['f1-score']:.4f}",
            f"  - Macro Avg   : Precision = {mb['classification_report']['macro avg']['precision']:.4f}, Recall = {mb['classification_report']['macro avg']['recall']:.4f}, F1 = {mb['classification_report']['macro avg']['f1-score']:.4f}",
            "-" * 78,
            "5. DIAGNOSTIC OBSERVATIONS & CAVEATS",
            "-" * 78,
            "1. Realistic Baseline Discrimination:",
            f"   - On unseen reviewers, the text-only TF-IDF baseline achieves an ROC-AUC of {res['roc_auc']:.4f}",
            f"     and PR-AUC of {res['pr_auc']:.4f}. At threshold 0.50, F1 is {m05['f1_score']:.4f}.",
            "   - Because the data was split with strict reviewer and template-cluster isolation,",
            "     the model cannot cheat by memorizing specific reviewers or leaking near-identical templates.",
            "",
            "2. Synthetic Dataset Structural Artifacts:",
            "   - As audited in Steps 2 and 3, 296 template clusters contain mixed labels where near-identical",
            "     phrasing was assigned to both genuine and fake classes during synthetic data generation.",
            "   - A linear/bag-of-words model like TF-IDF naturally struggles to separate identical phrasing",
            "     that carries opposing ground-truth labels based solely on n-grams.",
            "",
            "3. Next Modeling Steps (DeBERTa & Multimodal Fusion):",
            "   - Deep contextual transformers (DeBERTa-v3) will better capture contextual nuances.",
            "   - Multimodal fusion incorporating legitimate behavioral metadata (e.g. verified_purchase,",
            "     properly calculated historical rates) is expected to significantly boost discrimination.",
            "=" * 78,
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def run(self) -> None:
        """Execute full training and evaluation workflow."""
        df_train, df_val = self.load_data()
        X_train_tfidf, X_val_tfidf, y_train, y_val = self.extract_features(df_train, df_val)
        self.train_model(X_train_tfidf, y_train)
        self.evaluate_validation(X_val_tfidf, y_val)
        self.save_artifacts_and_report()
        print("\n[OK] Step 5 Baseline Training Pipeline successfully completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + LightGBM Baseline Model for Fake Review Detection v2."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/splits/train.csv",
        help="Path to training set CSV.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/splits/validation.csv",
        help="Path to validation set CSV.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="training/artifacts/baseline",
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory to save validation report.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Number of boosting iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    trainer = BaselineTrainer(
        train_path=args.train_path,
        val_path=args.val_path,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        max_features=args.max_features,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    trainer.run()


if __name__ == "__main__":
    main()
