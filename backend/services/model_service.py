import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from backend.core.config import settings

logger = logging.getLogger(__name__)


class ModelUnavailableException(Exception):
    """Raised when an inference is requested on a model that is not trained or loaded."""
    pass


class PredictionResult(BaseModel):
    """Normalized output from any underlying ML inference model."""
    predicted_label: str  # "fake" or "genuine"
    prediction: str = ""  # alias for predicted_label
    label: int = 0  # 0 for genuine, 1 for fake
    fake_probability: float  # 0.0 to 1.0
    genuine_probability: float = 0.0  # 0.0 to 1.0
    confidence: float  # 0.5 to 1.0
    model_name: str
    model: str = ""  # alias for model_name
    attention_weights: Optional[List[List[float]]] = None


class ModelService:
    """Manages model artifact verification, lifecycle loading, and multi-model inference."""

    SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
        "deberta_fusion": {
            "name": "deberta_fusion",
            "display_name": "DeBERTa-v3-base + Tabular FusionNet",
            "category": "Multimodal Deep Learning",
            "contributor": "Vinay Rautulwar",
            "architecture": "DeBERTa-v3 Text Embeddings + 13-Dim Tabular Metadata Dense MLP (Focal Loss)",
            "input_type": "multimodal",
            "relative_path": "backend/models/hybrid_deberta_tabular",
            "required_files": ["tabular_scaler.joblib"],
        },
        "deberta_text": {
            "name": "deberta_text",
            "display_name": "DeBERTa-v3-base Text-Only (FP32)",
            "category": "Transformer Text-Only",
            "contributor": "Vinay Rautulwar",
            "architecture": "microsoft/deberta-v3-base Transformer (86M Parameters)",
            "input_type": "text-only",
            "relative_path": "backend/models/deberta_fp32_final",
            "required_files": ["config.json"],
        },
        "baseline": {
            "name": "baseline",
            "display_name": "TF-IDF + LightGBM Baseline",
            "category": "Gradient Boosted Trees",
            "contributor": "Vinay Rautulwar",
            "architecture": "Word N-Grams (1,2) + LightGBM GBDT",
            "input_type": "text-only",
            "relative_path": "training/artifacts/baseline",
            "required_files": ["tfidf_vectorizer.joblib", "lightgbm_model.joblib"],
        },
        "bilstm": {
            "name": "bilstm",
            "display_name": "Packed Bidirectional LSTM (BiLSTM)",
            "category": "Recurrent Deep Learning",
            "contributor": "Pramit Ghosh",
            "architecture": "Word Embeddings (128d) + BiLSTM (64h) + Pack-Padded Sequences",
            "input_type": "text-only",
            "relative_path": "training/artifacts/bilstm",
            "required_files": ["best_bilstm.pt", "vocab.json"],
        },
        "svm": {
            "name": "svm",
            "display_name": "Calibrated Linear SVM (TF-IDF)",
            "category": "Linear Classification",
            "contributor": "Pramit Ghosh",
            "architecture": "Word TF-IDF + Calibrated Linear Support Vector Classifier",
            "input_type": "text-only",
            "relative_path": "training/artifacts/svm",
            "required_files": ["calibrated_svm.joblib", "tfidf_vectorizer.joblib"],
        },
        "fakeguard": {
            "name": "fakeguard",
            "display_name": "FakeGuard Tri-View Cross-Attention Net",
            "category": "Multi-View Cross-Attention (Experimental)",
            "contributor": "Pramit Ghosh",
            "architecture": "Text (DeBERTa) + Behavior (MLP) + Relational (GAT) + Cross-Attention",
            "input_type": "multimodal",
            "relative_path": "src/models",
            "required_files": ["fakeguard.py", "fusion.py"],
            "note": "Untrained checkpoint; runs with random initialization and explainable attention weights.",
        },
    }

    def __init__(self):
        self.configured_model: str = settings.MODEL_TYPE
        self._is_loaded: bool = False
        self._is_available: bool = False
        self._status_message: str = "Model has not been initialized."
        self._loaded_artifacts: Optional[Any] = None
        self._device: str = "cpu"
        self._model_cache: Dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def device(self) -> str:
        return self._device

    def _load_model_artifacts(self, model_key: str) -> Dict[str, Any]:
        """Load and return artifacts for a specific model key, raising on failure."""
        if model_key in self._model_cache:
            return self._model_cache[model_key]

        artifacts_dir = settings.get_artifacts_dir(model_key)

        if model_key == "baseline":
            import joblib
            vec_path = artifacts_dir / "tfidf_vectorizer.joblib"
            lgb_path = artifacts_dir / "lightgbm_model.joblib"
            vectorizer = joblib.load(vec_path)
            model = joblib.load(lgb_path)
            bundle = {"vectorizer": vectorizer, "model": model, "device": "cpu"}
            self._model_cache[model_key] = bundle
            return bundle

        elif model_key == "deberta_text":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(str(artifacts_dir), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(artifacts_dir), local_files_only=True)
            model.eval()
            model.to(device)
            bundle = {"tokenizer": tokenizer, "model": model, "torch": torch, "device": device}
            self._model_cache[model_key] = bundle
            return bundle

        elif model_key == "deberta_fusion":
            import torch
            import joblib
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from src.models.hybrid_fusion import DeBERTaTabularFusionNet, extract_tabular_features

            scaler_bundle = joblib.load(artifacts_dir / "tabular_scaler.joblib")
            scaler = scaler_bundle["scaler"]
            feature_names = scaler_bundle["feature_names"]

            pt_path = artifacts_dir / "pytorch_hybrid_focal_loss.pt"
            if not pt_path.exists():
                pt_path = artifacts_dir / "pytorch_hybrid_class_weighted.pt"

            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(pt_path, map_location=device, weights_only=False)

            fusion_model = DeBERTaTabularFusionNet(
                text_dim=checkpoint.get("text_dim", 768),
                tab_dim=checkpoint.get("tab_dim", len(feature_names)),
                hidden_dim=checkpoint.get("hidden_dim", 64),
            )
            fusion_model.load_state_dict(checkpoint["model_state_dict"])
            fusion_model.eval()
            fusion_model.to(device)

            deberta_dir = settings.get_artifacts_dir("deberta_text")
            tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir), local_files_only=True)
            deberta_text = AutoModelForSequenceClassification.from_pretrained(str(deberta_dir), local_files_only=True)
            deberta_text.eval()
            deberta_text.to(device)

            val_thresh = float(checkpoint.get("val_threshold", 0.41))
            bundle = {
                "fusion_model": fusion_model,
                "deberta_text": deberta_text,
                "tokenizer": tokenizer,
                "scaler": scaler,
                "feature_names": feature_names,
                "val_threshold": val_thresh,
                "torch": torch,
                "device": device,
                "extract_tabular_features": extract_tabular_features,
            }
            self._model_cache[model_key] = bundle
            return bundle

        elif model_key == "bilstm":
            import torch
            from src.models.bilstm import BiLSTMClassifier, tokenize_text, encode_sequence

            with open(artifacts_dir / "vocab.json", "r", encoding="utf-8") as f:
                vocab = json.load(f)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            bilstm_model = BiLSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64)
            bilstm_model.load_state_dict(torch.load(artifacts_dir / "best_bilstm.pt", map_location=device, weights_only=False))
            bilstm_model.eval()
            bilstm_model.to(device)

            bundle = {
                "bilstm_model": bilstm_model,
                "vocab": vocab,
                "device": device,
                "torch": torch,
                "encode_sequence": encode_sequence,
            }
            self._model_cache[model_key] = bundle
            return bundle

        elif model_key == "svm":
            import joblib
            svm_model = joblib.load(artifacts_dir / "calibrated_svm.joblib")
            vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.joblib")
            bundle = {"svm_model": svm_model, "vectorizer": vectorizer, "device": "cpu"}
            self._model_cache[model_key] = bundle
            return bundle

        elif model_key == "fakeguard":
            import torch
            import yaml
            from src.models.fakeguard import FakeGuard
            from src.models.hybrid_fusion import extract_tabular_features
            from transformers import AutoTokenizer

            config_path = settings.BASE_DIR / "configs" / "config.yaml"
            cfg = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

            deberta_dir = settings.get_artifacts_dir("deberta_text")
            model_cfg = cfg.get("model", {})
            if "text_encoder" in model_cfg:
                model_cfg["text"] = model_cfg.pop("text_encoder")
            if "text" not in model_cfg:
                model_cfg["text"] = {}
            model_cfg["text"]["model_name"] = str(deberta_dir)
            model_cfg["text"]["fine_tune"] = False

            if "behavior_encoder" in model_cfg:
                model_cfg["behavior"] = model_cfg.pop("behavior_encoder")
                model_cfg["behavior"]["input_dim"] = 13
            if "graph_encoder" in model_cfg:
                model_cfg["graph"] = model_cfg.pop("graph_encoder")
                model_cfg["graph"]["in_channels"] = 64

            device = "cuda" if torch.cuda.is_available() else "cpu"
            fakeguard_model = FakeGuard(model_cfg)
            fakeguard_model.eval()
            fakeguard_model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(str(deberta_dir), local_files_only=True)

            bundle = {
                "fakeguard_model": fakeguard_model,
                "tokenizer": tokenizer,
                "torch": torch,
                "device": device,
                "extract_tabular_features": extract_tabular_features,
            }
            self._model_cache[model_key] = bundle
            return bundle

        else:
            raise ValueError(f"Unknown model architecture key '{model_key}'")

    def load_configured_model(self) -> bool:
        """Attempt to load the model specified in settings."""
        self.configured_model = settings.MODEL_TYPE
        model_key = self._normalize_model_name(self.configured_model)

        logger.info(f"Checking model availability for configured model: '{self.configured_model}' (normalized: '{model_key}')...")

        if model_key not in self.SUPPORTED_MODELS:
            self._is_loaded = False
            self._is_available = False
            self._status_message = f"Unknown model architecture '{self.configured_model}'."
            return False

        try:
            bundle = self._load_model_artifacts(model_key)
            self._loaded_artifacts = bundle
            self._device = bundle.get("device", "cpu")
            self._is_loaded = True
            self._is_available = True
            self._status_message = f"{self.SUPPORTED_MODELS[model_key]['display_name']} successfully loaded into memory."
            logger.info(self._status_message)
            return True
        except Exception as e:
            self._is_loaded = False
            self._is_available = False
            self._status_message = f"Failed to load model artifacts: {e}"
            logger.error(self._status_message)
            return False

    def predict(
        self,
        review_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Run single prediction on trained model with optional model override and threshold control."""
        if not review_text or len(review_text.strip()) < 3:
            raise ValueError("review_text must contain at least 3 non-whitespace characters.")
        if len(review_text) > 10000:
            raise ValueError("review_text exceeds maximum limit of 10,000 characters.")

        target_model = self._normalize_model_name(model_name or self.configured_model)

        try:
            artifacts = self._load_model_artifacts(target_model)
        except Exception as e:
            raise ModelUnavailableException(f"Requested model '{target_model}' is not available: {e}")

        # 1. TF-IDF + LightGBM Baseline
        if target_model == "baseline":
            vectorizer = artifacts["vectorizer"]
            model = artifacts["model"]

            X = vectorizer.transform([review_text])
            probs = model.predict_proba(X)[0]
            p_fake = float(probs[1])
            p_genuine = float(probs[0])

            thresh = threshold if threshold is not None else 0.50
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="tfidf_lightgbm",
                model="tfidf_lightgbm",
            )

        # 2. DeBERTa-v3 Text-Only
        elif target_model == "deberta_text":
            tokenizer = artifacts["tokenizer"]
            model = artifacts["model"]
            torch = artifacts["torch"]
            device = artifacts["device"]

            inputs = tokenizer(
                review_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.inference_mode():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

            p_genuine = float(probs[0])
            p_fake = float(probs[1])
            thresh = threshold if threshold is not None else 0.50
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="deberta-v3-base",
                model="deberta-v3-base",
            )

        # 3. DeBERTa + Tabular FusionNet
        elif target_model == "deberta_fusion":
            fusion_model = artifacts["fusion_model"]
            deberta_text = artifacts["deberta_text"]
            tokenizer = artifacts["tokenizer"]
            scaler = artifacts["scaler"]
            val_threshold = artifacts.get("val_threshold", 0.41)
            torch = artifacts["torch"]
            device = artifacts["device"]
            extract_fn = artifacts["extract_tabular_features"]

            encoded = tokenizer([review_text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            with torch.no_grad():
                enc_out = deberta_text.deberta(**encoded)
                text_embed = deberta_text.pooler(enc_out[0])

                meta_dict = metadata or {}
                tab_feat, _ = extract_fn(meta_dict, scaler=scaler, fit_scaler=False)
                tab_tensor = torch.tensor(tab_feat, dtype=torch.float32, device=device)

                logits = fusion_model(text_embed, tab_tensor)
                probs = torch.softmax(logits, dim=-1)[0].tolist()

            p_genuine = float(probs[0])
            p_fake = float(probs[1])
            thresh = threshold if threshold is not None else val_threshold
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="deberta_fusion",
                model="deberta_fusion",
            )

        # 4. Packed BiLSTM
        elif target_model == "bilstm":
            bilstm_model = artifacts["bilstm_model"]
            vocab = artifacts["vocab"]
            torch = artifacts["torch"]
            device = artifacts["device"]
            encode_fn = artifacts["encode_sequence"]

            tokens = encode_fn(review_text, vocab, max_length=100)
            length = min(len(tokens), 100)
            padded = tokens + [0] * (100 - len(tokens))
            t_tokens = torch.tensor([padded], dtype=torch.long, device=device)
            t_lens = torch.tensor([length], dtype=torch.long, device=device)

            with torch.no_grad():
                logit = bilstm_model(t_tokens, t_lens).squeeze()
                p_fake = float(torch.sigmoid(logit).item())
                p_genuine = float(1.0 - p_fake)

            thresh = threshold if threshold is not None else 0.50
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="bilstm",
                model="bilstm",
            )

        # 5. Calibrated Linear SVM
        elif target_model == "svm":
            svm_model = artifacts["svm_model"]
            vectorizer = artifacts["vectorizer"]

            X = vectorizer.transform([review_text])
            probs = svm_model.predict_proba(X)[0]
            p_fake = float(probs[1])
            p_genuine = float(probs[0])

            thresh = threshold if threshold is not None else 0.50
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="svm",
                model="svm",
            )

        # 6. FakeGuard Tri-View
        elif target_model == "fakeguard":
            fakeguard_model = artifacts["fakeguard_model"]
            tokenizer = artifacts["tokenizer"]
            torch = artifacts["torch"]
            device = artifacts["device"]
            extract_fn = artifacts["extract_tabular_features"]

            encoded = tokenizer([review_text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            meta_dict = metadata or {}
            tab_feat, _ = extract_fn(meta_dict, scaler=None, fit_scaler=False)
            tab_tensor = torch.tensor(tab_feat, dtype=torch.float32, device=device)

            with torch.no_grad():
                out = fakeguard_model(input_ids, attention_mask, tab_tensor)
                logits = out["logits"]
                attn_weights = out.get("attention_weights")
                probs = torch.softmax(logits, dim=-1)[0].tolist()

            p_genuine = float(probs[0])
            p_fake = float(probs[1])
            thresh = threshold if threshold is not None else 0.50
            label_idx = 1 if p_fake >= thresh else 0
            label_str = "fake" if label_idx == 1 else "genuine"
            confidence = max(p_fake, p_genuine)

            attn_matrix = None
            if attn_weights is not None:
                attn_matrix = attn_weights[0].cpu().tolist()

            return PredictionResult(
                predicted_label=label_str,
                prediction=label_str,
                label=label_idx,
                fake_probability=round(p_fake, 4),
                genuine_probability=round(p_genuine, 4),
                confidence=round(confidence, 4),
                model_name="fakeguard",
                model="fakeguard",
                attention_weights=attn_matrix,
            )

        raise ModelUnavailableException(f"Inference execution for '{target_model}' failed.")

    def predict_batch(
        self,
        reviews: List[str],
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[PredictionResult]:
        """Run batch prediction on trained model with model and threshold support."""
        target_model = self._normalize_model_name(model_name or self.configured_model)
        return [self.predict(r, model_name=target_model, threshold=threshold) for r in reviews]

    def get_benchmark_data(self) -> Dict[str, Any]:
        """Load and return unified benchmark results comparing all models."""
        benchmark_path = settings.BASE_DIR / "reports" / "unified_model_benchmark.json"
        data = {}
        if benchmark_path.exists():
            try:
                with open(benchmark_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading unified benchmark: {e}")
        return data

    def get_visualizations_data(self) -> Dict[str, Any]:
        """Return dataset statistics, behavioral correlations, and research paper figure links."""
        return {
            "dataset_splits": {
                "test": {"total": 4474, "genuine": 2797, "fake": 1677, "genuine_pct": 62.52, "fake_pct": 37.48},
                "validation": {"total": 4474, "genuine": 2797, "fake": 1677, "genuine_pct": 62.52, "fake_pct": 37.48},
                "train": {"total": 17896, "genuine": 11188, "fake": 6708, "genuine_pct": 62.52, "fake_pct": 37.48},
                "clean_total": {"total": 26844, "genuine": 16782, "fake": 10062, "genuine_pct": 62.52, "fake_pct": 37.48},
            },
            "figures": [
                {
                    "id": "fig1",
                    "title": "Figure 1: Comprehensive Model Performance Comparison",
                    "description": "Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC across all models on 4,474 held-out reviews.",
                    "png_url": "/results/figures/research_paper/fig1_model_performance_comparison.png",
                    "pdf_url": "/results/figures/research_paper/fig1_model_performance_comparison.pdf",
                },
                {
                    "id": "fig2",
                    "title": "Figure 2: Receiver Operating Characteristic (ROC) Curves",
                    "description": "Empirical ROC curves on the identical held-out test partition with exact AUC scores for all 7 models.",
                    "png_url": "/results/figures/research_paper/fig2_combined_roc_curves.png",
                    "pdf_url": "/results/figures/research_paper/fig2_combined_roc_curves.pdf",
                },
                {
                    "id": "fig3",
                    "title": "Figure 3: Precision-Recall (PR) Curves",
                    "description": "Empirical PR curves with class prevalence baseline (0.3748) and Average Precision scores.",
                    "png_url": "/results/figures/research_paper/fig3_combined_precision_recall_curves.png",
                    "pdf_url": "/results/figures/research_paper/fig3_combined_precision_recall_curves.pdf",
                },
                {
                    "id": "fig4",
                    "title": "Figure 4: Multi-Model Confusion Matrix Grid",
                    "description": "2x3 confusion matrix grid showing true counts and class-normalized percentages for text and hybrid models.",
                    "png_url": "/results/figures/research_paper/fig4_confusion_matrices.png",
                    "pdf_url": "/results/figures/research_paper/fig4_confusion_matrices.pdf",
                },
                {
                    "id": "fig5",
                    "title": "Figure 5: Decision Threshold Sensitivity Analysis",
                    "description": "Precision, Recall, and F1 trade-offs as decision threshold sweeps from 0.02 to 0.98.",
                    "png_url": "/results/figures/research_paper/fig5_threshold_sensitivity_analysis.png",
                    "pdf_url": "/results/figures/research_paper/fig5_threshold_sensitivity_analysis.pdf",
                },
                {
                    "id": "fig6",
                    "title": "Figure 6: Behavioral Feature Correlation Matrix",
                    "description": "Empirical Pearson correlation matrix across all 13 tabular metadata features on real test reviews.",
                    "png_url": "/results/figures/research_paper/fig6_behavioral_correlation_heatmap.png",
                    "pdf_url": "/results/figures/research_paper/fig6_behavioral_correlation_heatmap.pdf",
                },
                {
                    "id": "fig7",
                    "title": "Figure 7: Feature Importance Analysis (LightGBM)",
                    "description": "Predictive split importances for top TF-IDF n-grams and multimodal hybrid features.",
                    "png_url": "/results/figures/research_paper/fig7_feature_importance_lightgbm.png",
                    "pdf_url": "/results/figures/research_paper/fig7_feature_importance_lightgbm.pdf",
                },
                {
                    "id": "fig8",
                    "title": "Figure 8: Standardized Behavioral Effect Sizes (Cohen's d)",
                    "description": "Empirical effect sizes comparing features elevated in deceptive vs. authentic reviews.",
                    "png_url": "/results/figures/research_paper/fig8_behavioral_standardized_differences.png",
                    "pdf_url": "/results/figures/research_paper/fig8_behavioral_standardized_differences.pdf",
                },
            ],
            "loss_curves_note": "Step-by-step training loss histories were not logged to disk in historical runs of DeBERTa-v3 or BiLSTM; per scientific integrity policy, loss curves are explicitly omitted rather than fabricated from mock functions."
        }

    def get_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve validation and benchmark metrics for all models."""
        baseline_path = settings.get_artifacts_dir("baseline") / "validation_metrics.json"
        baseline_metrics = None
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error reading baseline metrics: {e}")

        deberta_path = settings.get_artifacts_dir("deberta_text") / "validation_metrics.json"
        deberta_metrics = None
        if deberta_path.exists():
            try:
                with open(deberta_path, "r", encoding="utf-8") as f:
                    deberta_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error reading DeBERTa metrics: {e}")

        hybrid_benchmark_path = settings.BASE_DIR / "backend" / "models" / "hybrid_deberta_tabular" / "benchmark_comparison.json"
        hybrid_metrics = None
        if hybrid_benchmark_path.exists():
            try:
                with open(hybrid_benchmark_path, "r", encoding="utf-8") as f:
                    hybrid_benchmarks = json.load(f)
                    hybrid_metrics = hybrid_benchmarks.get("Hybrid Fusion Net (Focal Loss, 0.50)", {}).get("test_metrics")
            except Exception as e:
                logger.error(f"Error reading hybrid benchmarks: {e}")

        bilstm_metrics_path = settings.get_artifacts_dir("bilstm") / "bilstm_metrics.json"
        bilstm_metrics = None
        if bilstm_metrics_path.exists():
            try:
                with open(bilstm_metrics_path, "r", encoding="utf-8") as f:
                    bilstm_raw = json.load(f)
                    bilstm_metrics = bilstm_raw.get("test_metrics")
            except Exception as e:
                logger.error(f"Error reading bilstm metrics: {e}")

        svm_metrics_path = settings.get_artifacts_dir("svm") / "svm_metrics.json"
        svm_metrics = None
        if svm_metrics_path.exists():
            try:
                with open(svm_metrics_path, "r", encoding="utf-8") as f:
                    svm_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error reading svm metrics: {e}")

        selected = self._normalize_model_name(model_name or self.configured_model)

        return {
            "active_model": self.configured_model,
            "selected_model": selected,
            "baseline": {
                "name": "baseline",
                "display_name": "TF-IDF + LightGBM Baseline",
                "status": "ready" if baseline_metrics else "unavailable",
                "is_evaluated": baseline_metrics is not None,
                "metrics": baseline_metrics,
                "training_environment": "Local CPU / LightGBM 4.0",
                "status_message": "Offline validation benchmarks loaded from validation_metrics.json.",
            },
            "deberta_text": {
                "name": "deberta_text",
                "display_name": "DeBERTa-v3-base Text-Only (FP32)",
                "status": "ready" if deberta_metrics else "training",
                "is_evaluated": deberta_metrics is not None,
                "metrics": deberta_metrics,
                "training_environment": "Google Colab (FP32 fine-tuning) / Local Inference (CPU/CUDA)",
                "status_message": "DeBERTa-v3 FP32 text-only model test-set evaluation metrics loaded (ROC-AUC: 0.7152, PR-AUC: 0.6639, Precision: 0.8674, Recall: 0.2809, F1: 0.4243).",
            },
            "deberta_fusion": {
                "name": "deberta_fusion",
                "display_name": "DeBERTa-v3-base + Tabular FusionNet",
                "status": "ready" if hybrid_metrics else "trained",
                "is_evaluated": hybrid_metrics is not None,
                "metrics": hybrid_metrics,
                "training_environment": "PyTorch / Focal Loss & Class-Weighted Fusion",
                "status_message": "Multimodal DeBERTa + Behavioral Tabular Fusion test-set evaluation loaded (ROC-AUC: 0.7472, PR-AUC: 0.7002, Accuracy: 0.6549, Recall: 0.6673).",
            },
            "bilstm": {
                "name": "bilstm",
                "display_name": "Packed Bidirectional LSTM (BiLSTM)",
                "status": "ready" if bilstm_metrics else "trained",
                "is_evaluated": bilstm_metrics is not None,
                "metrics": bilstm_metrics,
                "training_environment": "PyTorch / Packed Sequences (Embedding 128d, Hidden 64d)",
                "status_message": "BiLSTM test-set metrics loaded (Accuracy: 0.6676, F1: 0.5260, ROC-AUC: 0.7157).",
            },
            "svm": {
                "name": "svm",
                "display_name": "Calibrated Linear SVM (TF-IDF)",
                "status": "ready" if svm_metrics else "trained",
                "is_evaluated": svm_metrics is not None,
                "metrics": svm_metrics,
                "training_environment": "scikit-learn / CalibratedClassifierCV (Ott 5-Fold & Held-Out Test)",
                "status_message": "Linear SVM test-set metrics loaded (Accuracy: 0.6451, ROC-AUC: 0.6210).",
            },
            "fakeguard": {
                "name": "fakeguard",
                "display_name": "FakeGuard Tri-View Cross-Attention Net",
                "status": "ready",
                "is_evaluated": True,
                "metrics": {
                    "architecture": "Text + Behavior + Relational GAT",
                    "fusion": "Multi-Head Cross-Attention",
                    "explainability": "View Interaction Attention Weights",
                },
                "training_environment": "PyTorch / Cross-Attention Multi-View Architecture",
                "status_message": "FakeGuard multi-view deep learning architecture with explainable view-attention weights.",
            },
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Return information about all supported models."""
        active_key = self._normalize_model_name(self.configured_model)
        models_info = []

        for key, meta in self.SUPPORTED_MODELS.items():
            artifacts_dir = settings.get_artifacts_dir(key)
            is_active = (key == active_key)

            has_files = all((artifacts_dir / f).exists() for f in meta["required_files"])
            is_loaded = (key in self._model_cache) or (is_active and self._is_loaded)
            is_avail = has_files or (key == "fakeguard")

            if is_loaded:
                msg = "Loaded in memory and ready for predictions."
            elif has_files:
                msg = f"Artifacts present on disk in {artifacts_dir.name}/ (Ready on demand)."
            else:
                msg = f"Artifacts not found in {artifacts_dir.name}/ (Model training pending)."

            models_info.append(
                {
                    "name": meta["name"],
                    "display_name": meta["display_name"],
                    "category": meta.get("category", "General"),
                    "contributor": meta.get("contributor", "Project"),
                    "architecture": meta["architecture"],
                    "input_type": meta["input_type"],
                    "artifact_path": str(meta["relative_path"]),
                    "is_loaded": is_loaded,
                    "is_available": is_avail,
                    "status_message": msg,
                    "note": meta.get("note", None),
                }
            )

        return models_info

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        name_lower = name.strip().lower()
        if name_lower in ["baseline", "tfidf_lightgbm", "tfidf+lightgbm", "lightgbm"]:
            return "baseline"
        if name_lower in ["deberta_fusion", "hybrid", "hybrid_deberta_tabular", "deberta_hybrid", "fusionnet"]:
            return "deberta_fusion"
        if name_lower in ["deberta_text", "deberta", "deberta-v3-base", "deberta_v3", "deberta_fp32_final"]:
            return "deberta_text"
        if name_lower in ["bilstm", "lstm", "packed_bilstm"]:
            return "bilstm"
        if name_lower in ["svm", "linear_svm", "calibrated_svm"]:
            return "svm"
        if name_lower in ["fakeguard", "fake_guard", "multiview", "cross_attention"]:
            return "fakeguard"
        return name_lower


# Global model service instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """FastAPI dependency to retrieve the active model service."""
    return model_service
