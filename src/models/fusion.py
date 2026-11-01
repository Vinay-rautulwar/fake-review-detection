import torch
import torch.nn as nn
from typing import Tuple

class CrossAttentionFusion(nn.Module):
    """
    Novel cross-attention module that dynamically fuses multi-view representations.
    
    This module takes embeddings from text, behavioral, and graph encoders,
    projects them to a shared space, and applies multi-head self-attention
    across the views to learn inter-view interactions. It outputs the aggregated
    fused representation along with attention weights for explainability.
    """
    def __init__(self, text_dim: int = 768, behavior_dim: int = 128, graph_dim: int = 128, 
                 projected_dim: int = 128, num_heads: int = 4, dropout: float = 0.2, num_classes: int = 2):
        super(CrossAttentionFusion, self).__init__()
        self.projected_dim = projected_dim
        
        self.text_proj = nn.Linear(text_dim, projected_dim)
        self.behavior_proj = nn.Linear(behavior_dim, projected_dim)
        self.graph_proj = nn.Linear(graph_dim, projected_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=projected_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(projected_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(projected_dim, projected_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projected_dim // 2, num_classes)
        )

    def forward(self, text_emb: torch.Tensor, behavior_emb: torch.Tensor, graph_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses multi-view representations using cross-attention.
        
        Args:
            text_emb (torch.Tensor): Semantic text embedding, shape (batch, text_dim).
            behavior_emb (torch.Tensor): Behavioral embedding, shape (batch, behavior_dim).
            graph_emb (torch.Tensor): Relational graph embedding, shape (batch, graph_dim).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Logits for classification, shape (batch, num_classes).
                - Attention weights for explainability, shape (batch, num_views, num_views).
        """
        text_proj = self.text_proj(text_emb)
        behav_proj = self.behavior_proj(behavior_emb)
        graph_proj = self.graph_proj(graph_emb)
        
        stacked_views = torch.stack([text_proj, behav_proj, graph_proj], dim=1)
        
        attn_output, attn_weights = self.attention(stacked_views, stacked_views, stacked_views)
        
        fused_views = self.layer_norm(stacked_views + self.dropout(attn_output))
        
        aggregated = fused_views.mean(dim=1)
        
        logits = self.classifier(aggregated)
        
        return logits, attn_weights
