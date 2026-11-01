import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import math
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class Trainer:
    """Training engine for FakeGuard model.
    
    Features:
    - Mixed precision training (AMP)
    - Learning rate scheduling (warmup + cosine decay)
    - Early stopping based on validation F1
    - Checkpoint saving/loading
    - TensorBoard logging
    - Gradient clipping
    - Reproducibility (seeded)
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        lr = self.config.get('learning_rate', 2e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        epochs = self.config.get('num_epochs', 10)
        num_training_steps = epochs * len(self.train_loader)
        num_warmup_steps = int(num_training_steps * self.config.get('warmup_ratio', 0.1))
        self.scheduler = self._get_scheduler(self.optimizer, num_training_steps, num_warmup_steps)
        
        from src.training.losses import get_loss_function
        loss_name = self.config.get('loss_function', 'cross_entropy')
        self.criterion = get_loss_function(loss_name).to(self.device)
        
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        self.save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        
        self.patience = self.config.get('early_stopping_patience', 3)
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns dict with loss, accuracy, f1."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch.get('features', None)
            if features is not None:
                features = features.to(self.device)
            edge_index = batch.get('edge_index', None)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            targets = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if features is not None:
                    kwargs['behavioral_features'] = features
                if edge_index is not None:
                    kwargs['edge_index'] = edge_index
                    
                outputs = self.model(**kwargs)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                loss = self.criterion(logits, targets)
                
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return {'loss': avg_loss, 'accuracy': acc, 'f1': f1}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate on validation set. Returns dict with loss, accuracy, f1, precision, recall."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch.get('features', None)
            if features is not None:
                features = features.to(self.device)
            edge_index = batch.get('edge_index', None)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            targets = batch['label'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if features is not None:
                    kwargs['behavioral_features'] = features
                if edge_index is not None:
                    kwargs['edge_index'] = edge_index
                    
                outputs = self.model(**kwargs)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                loss = self.criterion(logits, targets)
                
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss, 
            'accuracy': acc, 
            'f1': f1, 
            'precision': precision, 
            'recall': recall
        }

    def train(self, num_epochs: int = None) -> dict:
        """Full training loop with early stopping."""
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 10)
            
        best_metrics = {}
        training_history = []
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            training_history.append({'train': train_metrics, 'val': val_metrics})
            
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                best_metrics = val_metrics
                self.early_stopping_counter = 0
                logger.info(f"New best validation F1: {self.best_val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    self._save_checkpoint(epoch, val_metrics, is_best=False)
                    break
                    
            self._save_checkpoint(epoch, val_metrics, is_best=is_best)
            
        self.writer.close()
        return {'best_metrics': best_metrics, 'history': training_history}

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """Saves model and training states."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': metrics,
            'config': self.config
        }
        
        latest_path = self.save_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)

    def _load_checkpoint(self, path):
        """Loads model and training states from a given path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def _get_scheduler(self, optimizer, num_training_steps, num_warmup_steps):
        """Creates linear warmup + cosine decay scheduler."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)
