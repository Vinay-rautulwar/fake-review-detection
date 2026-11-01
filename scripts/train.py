"""
================================================================================
FakeGuard: End-to-End Training Script
================================================================================
Usage:
    python scripts/train.py
    python scripts/train.py --debug
    python scripts/train.py --epochs 5
    python scripts/train.py --model-type text_only
================================================================================
"""
import argparse
import sys
import logging
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
from src.data.preprocessing import TextPreprocessor
from src.data.feature_engineering import BehavioralFeatureExtractor
from src.data.dataset import FakeReviewDataset
from src.models.fakeguard import FakeGuard
from src.training.losses import get_loss_function

logger = logging.getLogger(__name__)



class TextOnlyModel(nn.Module):
    """Ablation: Only uses the BERT text encoder + classifier head."""
    def __init__(self, config):
        super().__init__()
        from src.models.text_encoder import TextEncoder
        text_cfg = config.get('text', {})
        self.text_encoder = TextEncoder(
            model_name=text_cfg.get('model_name', 'bert-base-uncased'),
            hidden_dim=text_cfg.get('hidden_dim', 768),
            dropout=text_cfg.get('dropout', 0.1),
            fine_tune=text_cfg.get('fine_tune', True)
        )
        hidden = text_cfg.get('hidden_dim', 768)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, config.get('num_classes', 2))
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        text_emb = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(text_emb)
        return {'logits': logits}


class TextBehaviorModel(nn.Module):
    """Ablation: Text + Behavior encoders (no graph), simple concat fusion."""
    def __init__(self, config):
        super().__init__()
        from src.models.text_encoder import TextEncoder
        from src.models.behavior_encoder import BehaviorEncoder
        text_cfg = config.get('text', {})
        behav_cfg = config.get('behavior', {})

        self.text_encoder = TextEncoder(
            model_name=text_cfg.get('model_name', 'bert-base-uncased'),
            hidden_dim=text_cfg.get('hidden_dim', 768),
            dropout=text_cfg.get('dropout', 0.1),
            fine_tune=text_cfg.get('fine_tune', True)
        )
        self.behavior_encoder = BehaviorEncoder(
            input_dim=behav_cfg.get('input_dim', 16),
            hidden_dims=behav_cfg.get('hidden_dims', [128, 128]),
            dropout=behav_cfg.get('dropout', 0.3)
        )
        combined_dim = text_cfg.get('hidden_dim', 768) + behav_cfg.get('hidden_dims', [128, 128])[-1]
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(0.2),
            nn.Linear(combined_dim, config.get('num_classes', 2))
        )

    def forward(self, input_ids, attention_mask, behavioral_features=None, **kwargs):
        text_emb = self.text_encoder(input_ids, attention_mask)
        if behavioral_features is not None:
            behav_emb = self.behavior_encoder(behavioral_features)
            combined = torch.cat([text_emb, behav_emb], dim=-1)
        else:
            combined = text_emb
        logits = self.classifier(combined)
        return {'logits': logits}



def load_data(config, debug=False, data_path=None):
    """
    Loads or generates data, extracts features, tokenizes text, and returns DataLoaders.
    If data_path is provided, loads that CSV directly (for real datasets).
    """
    processed_file = Path(data_path) if data_path else PROJECT_ROOT / "data" / "processed" / "dataset_with_features.csv"
    raw_file = PROJECT_ROOT / "data" / "raw" / "dataset.csv"

    if processed_file.exists():
        logger.info(f"Loading dataset from: {processed_file}")
        df = pd.read_csv(processed_file)
    elif raw_file.exists() and not data_path:
        logger.info(f"Extracting features from raw dataset: {raw_file}")
        df_raw = pd.read_csv(raw_file)
        extractor = BehavioralFeatureExtractor(df_raw)
        df = extractor.extract_all_features()
    else:
        logger.info("No dataset found. Generating synthetic dataset...")
        from data.download_data import generate_sample_dataset
        df_raw = generate_sample_dataset(str(PROJECT_ROOT / "data" / "raw"), num_reviews=2000)
        extractor = BehavioralFeatureExtractor(df_raw)
        df = extractor.extract_all_features()

    preprocessor = TextPreprocessor()
    if 'cleaned_text' not in df.columns:
        text_col = 'text' if 'text' in df.columns else df.columns[0]
        df['cleaned_text'] = df[text_col].astype(str).apply(preprocessor.clean_text)

    id_cols = {'review_id', 'user_id', 'product_id', 'label', 'text', 'cleaned_text',
               'date', 'class_name', 'dataset', 'category', 'source', 'hotel', 'polarity',
               'deceptive', 'text_', 'reviewContent', 'flagged', 'name', 'location'}
    behavioral_cols = [
        c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in behavioral_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    if debug:
        logger.info("DEBUG mode: Using only 200 samples")
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

    scaler = StandardScaler()
    behav_array = scaler.fit_transform(df[behavioral_cols].values).astype(np.float32)

    texts = df['cleaned_text'].tolist()
    labels = df['label'].values.astype(np.int64)

    logger.info(f"Dataset: {len(df)} samples | {len(behavioral_cols)} behavioral features")
    logger.info(f"  Fake: {(labels == 1).sum()} | Genuine: {(labels == 0).sum()}")
    logger.info(f"  Behavioral features: {behavioral_cols}")

    from transformers import AutoTokenizer
    model_name = config.get('text', {}).get('model_name', 'bert-base-uncased')
    max_length = config.get('max_seq_length', 128)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = FakeReviewDataset(
        texts=texts,
        behavioral_features=behav_array,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )

    total = len(dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    batch_size = config.get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    logger.info(f"  Batch size: {batch_size} | Train batches: {len(train_loader)}")

    return train_loader, val_loader, test_loader, len(behavioral_cols)



def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        behavioral_features = batch['behavioral_features'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                behavioral_features=behavioral_features
            )
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(train_loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {'loss': avg_loss, 'accuracy': acc, 'f1': f1}


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, use_amp=False):
    """Validate on val/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        behavioral_features = batch['behavioral_features'].to(device)
        targets = batch['label'].to(device)

        with autocast('cuda', enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                behavioral_features=behavioral_features
            )
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, targets)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / max(len(val_loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_binary = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    try:
        auc_roc = roc_auc_score(all_targets, all_probs)
        auc_pr = average_precision_score(all_targets, all_probs)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0

    return {
        'loss': avg_loss, 'accuracy': acc, 'f1': f1, 'f1_binary': f1_binary,
        'precision': precision, 'recall': recall,
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'y_true': np.array(all_targets), 'y_pred': np.array(all_preds), 'y_prob': np.array(all_probs)
    }


def get_lr_scheduler(optimizer, num_training_steps, num_warmup_steps):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)



def create_model(model_type, config):
    """Factory to create model based on type (for ablation studies)."""
    if model_type == 'fakeguard':
        return FakeGuard(config)
    elif model_type == 'text_only':
        return TextOnlyModel(config)
    elif model_type == 'text_behavior':
        return TextBehaviorModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def main():
    parser = argparse.ArgumentParser(description='Train FakeGuard model')
    parser.add_argument('--debug', action='store_true', help='Quick debug run with small subset')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--model-type', type=str, default='fakeguard',
                        choices=['fakeguard', 'text_only', 'text_behavior'],
                        help='Model variant for ablation studies')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['cross_entropy', 'focal', 'label_smoothing'])
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to a custom processed CSV dataset')
    parser.add_argument('--results-prefix', type=str, default=None,
                        help='Prefix for results filenames (e.g. deceptive, fake40k, yelpchi)')
    args = parser.parse_args()

    yaml_config = {}
    try:
        yaml_config = load_config(str(PROJECT_ROOT / 'configs' / 'config.yaml'))
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}. Using defaults.")

    num_behav_features = 16

    config = {
        'text': {
            'model_name': yaml_config.get('model', {}).get('text_encoder', {}).get('name', 'bert-base-uncased'),
            'hidden_dim': yaml_config.get('model', {}).get('text_encoder', {}).get('hidden_dim', 768),
            'dropout': yaml_config.get('model', {}).get('text_encoder', {}).get('dropout', 0.1),
            'fine_tune': yaml_config.get('model', {}).get('text_encoder', {}).get('fine_tune', True),
        },
        'behavior': {
            'input_dim': num_behav_features,
            'hidden_dims': yaml_config.get('model', {}).get('behavior_encoder', {}).get('hidden_dims', [128, 128]),
            'dropout': yaml_config.get('model', {}).get('behavior_encoder', {}).get('dropout', 0.3),
        },
        'graph': {
            'in_channels': 64,
            'hidden_dim': yaml_config.get('model', {}).get('graph_encoder', {}).get('hidden_dim', 128),
            'num_heads': yaml_config.get('model', {}).get('graph_encoder', {}).get('num_heads', 4),
            'num_layers': yaml_config.get('model', {}).get('graph_encoder', {}).get('num_layers', 2),
            'dropout': yaml_config.get('model', {}).get('graph_encoder', {}).get('dropout', 0.2),
        },
        'fusion': {
            'projected_dim': yaml_config.get('model', {}).get('fusion', {}).get('projected_dim', 128),
            'num_heads': yaml_config.get('model', {}).get('fusion', {}).get('num_heads', 4),
            'dropout': yaml_config.get('model', {}).get('fusion', {}).get('dropout', 0.2),
            'num_classes': yaml_config.get('model', {}).get('num_classes', 2),
        },
        'num_classes': yaml_config.get('model', {}).get('num_classes', 2),
        'max_seq_length': yaml_config.get('data', {}).get('max_seq_length', 128),
        'batch_size': args.batch_size or (4 if args.debug else 8),
        'learning_rate': args.lr or yaml_config.get('training', {}).get('learning_rate', 2e-5),
        'weight_decay': yaml_config.get('training', {}).get('weight_decay', 0.01),
        'num_epochs': args.epochs or (3 if args.debug else yaml_config.get('training', {}).get('epochs', 10)),
        'warmup_ratio': 0.1,
        'use_amp': True,
        'loss_function': args.loss,
        'early_stopping_patience': yaml_config.get('training', {}).get('early_stopping_patience', 3),
        'save_dir': str(PROJECT_ROOT / 'results' / 'checkpoints'),
        'log_dir': str(PROJECT_ROOT / 'results' / 'logs'),
    }

    if args.debug:
        config['max_seq_length'] = 64

    setup_logging(config['log_dir'])
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print(f"  FakeGuard Training - Model: {args.model_type.upper()}")
    print(f"  Device: {device} | Epochs: {config['num_epochs']} | Batch: {config['batch_size']}")
    print(f"  LR: {config['learning_rate']} | Loss: {config['loss_function']}")
    print(f"  Debug: {args.debug}")
    print("=" * 70)

    train_loader, val_loader, test_loader, num_behav = load_data(config, debug=args.debug, data_path=args.data_path)

    config['behavior']['input_dim'] = num_behav
    logger.info(f"Updated behavior input_dim to {num_behav}")

    model = create_model(args.model_type, config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Parameters: {total_params:,} total | {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    num_training_steps = config['num_epochs'] * len(train_loader)
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    scheduler = get_lr_scheduler(optimizer, num_training_steps, num_warmup_steps)

    criterion = get_loss_function(config['loss_function']).to(device)
    scaler = GradScaler('cuda', enabled=config['use_amp'])

    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(save_dir / 'tb_logs'))

    best_val_f1 = 0.0
    patience_counter = 0
    patience = config['early_stopping_patience']
    training_history = []

    print(f"\n{'='*70}")
    print(f"  Starting Training Loop - {config['num_epochs']} epochs")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, use_amp=config['use_amp']
        )

        val_metrics = validate(
            model, val_loader, criterion, device, epoch, use_amp=config['use_amp']
        )

        epoch_time = time.time() - epoch_start

        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
        writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC-ROC/Val', val_metrics['auc_roc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc_roc']:.4f}")

        training_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_acc': val_metrics['accuracy'],
            'val_auc_roc': val_metrics['auc_roc'],
        })

        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            print(f"  [*] New best F1: {best_val_f1:.4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': {k: v for k, v in val_metrics.items() if k not in ('y_true', 'y_pred', 'y_prob')},
                'model_type': args.model_type,
            }, save_dir / 'checkpoint_best.pt')
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch} epochs.")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'metrics': {k: v for k, v in val_metrics.items() if k not in ('y_true', 'y_pred', 'y_prob')},
            'model_type': args.model_type,
        }, save_dir / 'checkpoint_latest.pt')

        print()

    total_time = time.time() - start_time
    writer.close()

    print("=" * 70)
    print("  Final Test Set Evaluation")
    print("=" * 70)

    best_ckpt_path = save_dir / 'checkpoint_best.pt'
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']})\n")

    test_metrics = validate(model, test_loader, criterion, device, epoch=0, use_amp=config['use_amp'])

    print(f"\n  Test Results ({args.model_type.upper()}):")
    print(f"    Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"    F1-Macro:    {test_metrics['f1']:.4f}")
    print(f"    F1-Binary:   {test_metrics['f1_binary']:.4f}")
    print(f"    Precision:   {test_metrics['precision']:.4f}")
    print(f"    Recall:      {test_metrics['recall']:.4f}")
    print(f"    AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    print(f"    AUC-PR:      {test_metrics['auc_pr']:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(
        test_metrics['y_true'], test_metrics['y_pred'],
        target_names=['Genuine (0)', 'Fake (1)'], digits=4
    ))

    results = {
        'model_type': args.model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_seconds': total_time,
        'num_epochs_trained': len(training_history),
        'best_val_f1': best_val_f1,
        'test_metrics': {k: float(v) for k, v in test_metrics.items() if k not in ('y_true', 'y_pred', 'y_prob')},
        'training_history': training_history,
        'config': {k: v for k, v in config.items() if k not in ('text', 'behavior', 'graph', 'fusion')},
    }

    prefix = f"{args.results_prefix}_{args.model_type}" if args.results_prefix else args.model_type
    results_path = PROJECT_ROOT / 'results' / f'{prefix}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")
    print(f"  Checkpoint saved to: {save_dir / 'checkpoint_best.pt'}")
    print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
