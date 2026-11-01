"""
FakeGuard: A Multi-View Deep Learning Framework with Explainable Attention for Fake Review Detection.

FakeGuard combines textual semantics (via Transformers), behavioral metadata features
(via multi-layer perceptrons), and relational graph structure (via Graph Neural Networks)
fused through an explainable multi-head cross-attention mechanism.
"""

__version__ = "0.1.0"
__author__ = "FakeGuard Research Team"

__all__ = [
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]
