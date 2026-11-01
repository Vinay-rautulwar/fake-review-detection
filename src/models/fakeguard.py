import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .text_encoder import TextEncoder
from .behavior_encoder import BehaviorEncoder
from .graph_encoder import GraphEncoder
from .fusion import CrossAttentionFusion

class FakeGuard(nn.Module):
    """
    FakeGuard: Multi-View Deep Learning Framework for Fake Review Detection.
    
    Combines three complementary views:
    1. Semantic View (BERT text encoder)
    2. Behavioral View (MLP behavioral encoder)  
    3. Relational View (GAT graph encoder)
    
    Fused via a novel cross-attention mechanism with built-in explainability.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FakeGuard model with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing hyperparameters for each sub-module.
                Expected keys: 'text', 'behavior', 'graph', 'fusion'.
        """
        super(FakeGuard, self).__init__()
        
        text_cfg = config.get('text', {})
        behav_cfg = config.get('behavior', {})
        graph_cfg = config.get('graph', {})
        fusion_cfg = config.get('fusion', {})
        
        self.text_encoder = TextEncoder(
            model_name=text_cfg.get('model_name', 'bert-base-uncased'),
            hidden_dim=text_cfg.get('hidden_dim', 768),
            dropout=text_cfg.get('dropout', 0.1),
            fine_tune=text_cfg.get('fine_tune', True)
        )
        
        self.behavior_encoder = BehaviorEncoder(
            input_dim=behav_cfg.get('input_dim', 15),
            hidden_dims=behav_cfg.get('hidden_dims', [128, 128]),
            dropout=behav_cfg.get('dropout', 0.3)
        )
        
        self.graph_encoder = GraphEncoder(
            in_channels=graph_cfg.get('in_channels', 64),
            hidden_dim=graph_cfg.get('hidden_dim', 128),
            num_heads=graph_cfg.get('num_heads', 4),
            num_layers=graph_cfg.get('num_layers', 2),
            dropout=graph_cfg.get('dropout', 0.2)
        )
        
        self.fusion = CrossAttentionFusion(
            text_dim=text_cfg.get('hidden_dim', 768),
            behavior_dim=behav_cfg.get('hidden_dims', [128, 128])[-1],
            graph_dim=graph_cfg.get('hidden_dim', 128),
            projected_dim=fusion_cfg.get('projected_dim', 128),
            num_heads=fusion_cfg.get('num_heads', 4),
            dropout=fusion_cfg.get('dropout', 0.2),
            num_classes=fusion_cfg.get('num_classes', 2)
        )
        
        self.last_attention_weights = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                behavioral_features: torch.Tensor, node_features: Optional[torch.Tensor] = None, 
                edge_index: Optional[torch.Tensor] = None, node_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the complete FakeGuard model.
        
        Args:
            input_ids (torch.Tensor): Text token IDs, shape (batch, seq_len).
            attention_mask (torch.Tensor): Text attention mask, shape (batch, seq_len).
            behavioral_features (torch.Tensor): Behavioral features, shape (batch, behav_dim).
            node_features (torch.Tensor, optional): Graph node features.
            edge_index (torch.Tensor, optional): Graph edge index.
            node_indices (torch.Tensor, optional): Indices of the current batch nodes in the graph.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'logits', 'attention_weights', 
                                     and view-specific embeddings.
        """
        batch_size = input_ids.size(0)
        
        text_emb = self.text_encoder(input_ids, attention_mask)
        
        behav_emb = self.behavior_encoder(behavioral_features)
        
        graph_dim = self.fusion.graph_proj.in_features
        if node_features is not None and edge_index is not None and node_indices is not None:
            full_graph_emb = self.graph_encoder(node_features, edge_index)
            if full_graph_emb is not None:
                graph_emb = full_graph_emb[node_indices]
            else:
                graph_emb = torch.zeros((batch_size, graph_dim), device=input_ids.device)
        else:
            graph_emb = torch.zeros((batch_size, graph_dim), device=input_ids.device)
            
        logits, attn_weights = self.fusion(text_emb, behav_emb, graph_emb)
        self.last_attention_weights = attn_weights
        
        return {
            'logits': logits,
            'attention_weights': attn_weights,
            'text_embedding': text_emb,
            'behavior_embedding': behav_emb,
            'graph_embedding': graph_emb
        }
        
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Returns the last computed attention weights for explainability.
        
        Returns:
            Optional[torch.Tensor]: Attention weights from the cross-attention module.
        """
        return self.last_attention_weights
