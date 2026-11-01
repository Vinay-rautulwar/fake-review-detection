import torch
import torch.nn as nn
from typing import Optional

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class GraphEncoder(nn.Module):
    """
    Graph Attention Network for modeling reviewer-product relationships.
    
    Learns node embeddings by aggregating information from neighbors using
    attention mechanisms. If graph data is not available, acts gracefully.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super(GraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        if not HAS_PYG:
            import warnings
            warnings.warn("torch_geometric is not installed. GraphEncoder will be a mock module.")
            self.layers = nn.ModuleList()
            return
            
        self.layers = nn.ModuleList()
        current_dim = in_channels
        
        for i in range(num_layers):
            concat = True if i < num_layers - 1 else False
            out_dim = hidden_dim // num_heads if concat else hidden_dim
            
            conv = GATConv(
                current_dim, 
                out_dim, 
                heads=num_heads, 
                concat=concat, 
                dropout=dropout
            )
            self.layers.append(conv)
            current_dim = hidden_dim

        self.activation = nn.ELU()

    def forward(self, x: Optional[torch.Tensor], edge_index: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Forward pass for the graph encoder.
        
        Args:
            x (torch.Tensor, optional): Node features, shape (num_nodes, in_channels).
            edge_index (torch.Tensor, optional): Graph connectivity, shape (2, num_edges).
            
        Returns:
            torch.Tensor: Node embeddings, shape (num_nodes, hidden_dim).
        """
        if x is None or edge_index is None or not hasattr(self, 'layers') or len(self.layers) == 0:
            return None
            
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                
        return x
