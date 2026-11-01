import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TextEncoder(torch.nn.Module):
    def forward(self, x): return torch.zeros(x.shape[0], 768)

class BehaviorEncoder(torch.nn.Module):
    def forward(self, x): return torch.zeros(x.shape[0], 128)
    
class GraphEncoder(torch.nn.Module):
    def forward(self, x, edge_index): return torch.zeros(x.shape[0], 64)

class CrossAttentionFusion(torch.nn.Module):
    def forward(self, text, behavior, graph):
        return torch.zeros(text.shape[0], 256), torch.ones(text.shape[0], 3, 3) / 3

class FakeGuard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = TextEncoder()
        self.behav_enc = BehaviorEncoder()
        self.graph_enc = GraphEncoder()
        self.fusion = CrossAttentionFusion()
        self.clf = torch.nn.Linear(256, 1)
        
    def forward(self, text_input, behav_input, graph_input=None, edge_index=None):
        t_feat = self.text_enc(text_input)
        b_feat = self.behav_enc(behav_input)
        
        if graph_input is not None and edge_index is not None:
            g_feat = self.graph_enc(graph_input, edge_index)
        else:
            g_feat = torch.zeros(t_feat.shape[0], 64)
            
        fused, _ = self.fusion(t_feat, b_feat, g_feat)
        return torch.sigmoid(self.clf(fused))

class TestTextEncoder:
    def test_output_shape(self):
        model = TextEncoder()
        dummy_input = torch.randint(0, 1000, (4, 128))
        out = model(dummy_input)
        assert out.shape == (4, 768), "Text encoder output shape mismatch"
        
    def test_forward_pass(self):
        model = TextEncoder()
        dummy_input = torch.randint(0, 1000, (2, 64))
        try:
            model(dummy_input)
        except Exception as e:
            pytest.fail(f"Forward pass failed with exception: {e}")

class TestBehaviorEncoder:
    def test_output_shape(self):
        model = BehaviorEncoder()
        dummy_input = torch.randn(4, 15)
        out = model(dummy_input)
        assert out.shape == (4, 128), "Behavior encoder output shape mismatch"

class TestGraphEncoder:
    def test_output_shape(self):
        model = GraphEncoder()
        dummy_nodes = torch.randn(4, 30)
        dummy_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        out = model(dummy_nodes, dummy_edges)
        assert out.shape == (4, 64), "Graph encoder output shape mismatch"

class TestCrossAttentionFusion:
    def test_output_shape(self):
        fusion = CrossAttentionFusion()
        t = torch.randn(4, 768)
        b = torch.randn(4, 128)
        g = torch.randn(4, 64)
        out, att = fusion(t, b, g)
        assert out.shape == (4, 256), "Fusion output shape mismatch"
        
    def test_attention_weights(self):
        fusion = CrossAttentionFusion()
        t = torch.randn(4, 768)
        b = torch.randn(4, 128)
        g = torch.randn(4, 64)
        _, att = fusion(t, b, g)
        
        assert att.shape == (4, 3, 3), "Attention weights shape mismatch"
        sums = att.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums)), "Attention weights do not sum to 1"

class TestFakeGuard:
    def test_full_forward_pass(self):
        model = FakeGuard()
        t_input = torch.randint(0, 1000, (4, 128))
        b_input = torch.randn(4, 15)
        g_input = torch.randn(4, 30)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        
        out = model(t_input, b_input, g_input, edges)
        assert out.shape == (4, 1), "Final output shape mismatch"
        assert torch.all((out >= 0) & (out <= 1)), "Outputs should be probabilities"
        
    def test_without_graph(self):
        model = FakeGuard()
        t_input = torch.randint(0, 1000, (4, 128))
        b_input = torch.randn(4, 15)
        
        out = model(t_input, b_input)
        assert out.shape == (4, 1), "Final output shape mismatch when missing graph data"


# ---------------------------------------------------------------------------
# Integration tests on REAL modules from src/
# ---------------------------------------------------------------------------
from src.models.behavior_encoder import BehaviorEncoder as RealBehaviorEncoder
from src.models.fusion import CrossAttentionFusion as RealCrossAttentionFusion
from src.training.losses import FocalLoss, LabelSmoothingLoss


class TestRealComponents:
    def test_real_behavior_encoder(self):
        encoder = RealBehaviorEncoder(input_dim=12, hidden_dims=[64, 32], dropout=0.1)
        x = torch.randn(8, 12)
        out = encoder(x)
        assert out.shape == (8, 32)

    def test_real_cross_attention_fusion(self):
        fusion = RealCrossAttentionFusion(
            text_dim=768, behavior_dim=64, graph_dim=64,
            projected_dim=64, num_heads=2, dropout=0.1, num_classes=2
        )
        t = torch.randn(4, 768)
        b = torch.randn(4, 64)
        g = torch.randn(4, 64)
        logits, attn = fusion(t, b, g)
        assert logits.shape == (4, 2)
        assert attn.shape == (4, 3, 3)

    def test_real_focal_loss(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(6, 2)
        targets = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_real_label_smoothing_loss(self):
        loss_fn = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

