import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    BERT/RoBERTa-based text encoder for review text.
    
    This module uses a pre-trained transformer model from HuggingFace to encode
    the raw review text into dense semantic representations.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', hidden_dim: int = 768, dropout: float = 0.1, fine_tune: bool = True):
        super(TextEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer = AutoModel.from_pretrained(model_name)
        
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the text encoder.
        
        Args:
            input_ids (torch.Tensor): Tensor of token ids, shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Tensor of attention masks, shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: [CLS] token embedding, shape (batch_size, hidden_dim).
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls_embedding)
