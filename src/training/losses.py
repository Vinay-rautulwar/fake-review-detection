import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in fake review detection.
    
    Focal loss down-weights easy examples and focuses on hard misclassified ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for class balance (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the focal loss."""
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing for regularization."""
    def __init__(self, num_classes: int = 2, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes label smoothed cross entropy."""
        log_probs = F.log_softmax(inputs, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
            
        loss = torch.sum(-true_dist * log_probs, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss by name.
    
    Supports: 'cross_entropy', 'focal', 'label_smoothing'
    """
    loss_name = loss_name.lower()
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
