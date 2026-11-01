"""
Model architectures and neural network modules for Fake Review Detection.

Provides:
- FakeGuard: End-to-end multi-view network combining text, behavior, and graph streams
- CrossAttentionFusion: Multi-head cross-attention view fusion
- TextEncoder: Transformer-based (BERT/RoBERTa/DeBERTa) semantic representation
- BehaviorEncoder: Deep MLP encoder for user and product behavioral metadata
- GraphEncoder: PyTorch Geometric GNN (GAT/GCN) for relational graphs
- DeBERTaTabularFusionNet: Multimodal hybrid network combining DeBERTa contextual text and tabular behavioral metadata
- extract_tabular_features: Leakage-isolated tabular metadata extraction function
"""

from .text_encoder import TextEncoder
from .behavior_encoder import BehaviorEncoder
from .graph_encoder import GraphEncoder
from .fusion import CrossAttentionFusion
from .fakeguard import FakeGuard
from .hybrid_fusion import DeBERTaTabularFusionNet, extract_tabular_features

__all__ = [
    "FakeGuard",
    "CrossAttentionFusion",
    "TextEncoder",
    "BehaviorEncoder",
    "GraphEncoder",
    "DeBERTaTabularFusionNet",
    "extract_tabular_features",
]
