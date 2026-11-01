"""
Module for PyTorch and PyTorch Geometric dataset definitions.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional, Any
import numpy as np
import pandas as pd

try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
except ImportError:
    torch_geometric = None
    Data = None
    HeteroData = None

class FakeReviewDataset(Dataset):
    """
    Standard PyTorch Dataset for Fake Review Detection.
    """
    def __init__(self, texts: List[str], behavioral_features: np.ndarray, labels: np.ndarray, tokenizer: Any, max_length: int = 256):
        """
        Args:
            texts: List of review texts.
            behavioral_features: Numpy array of behavioral features for each review.
            labels: Numpy array of labels (0=genuine, 1=fake).
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.behavioral_features = behavioral_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'behavioral_features': torch.tensor(self.behavioral_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_dataloaders(dataset: Dataset, batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits dataset and creates PyTorch DataLoaders.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if torch_geometric is not None:
    class GraphFakeReviewDataset(Dataset):
        """
        Dataset class that includes graph data structure for GNNs.
        """
        def __init__(self, df: pd.DataFrame, text_embeddings: torch.Tensor, behavioral_features: torch.Tensor, labels: torch.Tensor):
            self.df = df
            self.text_embeddings = text_embeddings
            self.behavioral_features = behavioral_features
            self.labels = labels
            
            self.graph_data = self._build_graph()

        def _build_graph(self) -> Data:
            """Builds a homogeneous graph representing the reviewer-product bipartite relationships."""
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
            return Data(x=self.behavioral_features, edge_index=edge_index, y=self.labels)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            return {
                'features': self.behavioral_features[idx],
                'label': self.labels[idx]
            }

    def build_heterogeneous_graph(df: pd.DataFrame, review_features: torch.Tensor, labels: torch.Tensor) -> 'HeteroData':
        """
        Helper function to build a heterogeneous graph from the dataframe.
        """
        data = HeteroData()
        
        user_ids = df['user_id'].unique()
        product_ids = df['product_id'].unique()
        
        user_mapping = {uid: i for i, uid in enumerate(user_ids)}
        product_mapping = {pid: i for i, pid in enumerate(product_ids)}
        
        data['user'].x = torch.zeros((len(user_ids), 1)) 
        data['product'].x = torch.zeros((len(product_ids), 1))
        data['review'].x = review_features
        data['review'].y = labels
        
        user_src = [user_mapping[uid] for uid in df['user_id']]
        review_dst = list(range(len(df)))
        data['user', 'writes', 'review'].edge_index = torch.tensor([user_src, review_dst], dtype=torch.long)
        
        review_src = list(range(len(df)))
        product_dst = [product_mapping[pid] for pid in df['product_id']]
        data['review', 'reviews', 'product'].edge_index = torch.tensor([review_src, product_dst], dtype=torch.long)
        
        return data
