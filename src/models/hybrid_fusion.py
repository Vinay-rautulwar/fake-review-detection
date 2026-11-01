import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List


def extract_tabular_features(df_or_dict, scaler=None, fit_scaler=False):
    """
    Extracts tabular metadata features with strict leakage protection.
    Accepts either a pandas DataFrame or a dict/list of dicts.
    """
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    elif isinstance(df_or_dict, list):
        df = pd.DataFrame(df_or_dict)
    else:
        df = df_or_dict.copy()

    # Provide safe defaults for any missing metadata columns
    defaults = {
        "rating": 5.0,
        "product_avg_rating": 4.0,
        "reviewer_avg_rating": 4.0,
        "verified_purchase": 1,
        "reviewer_review_count": 5,
        "days_since_previous_review": 30.0,
        "reviewer_account_age_days": 180.0,
        "reviewer_verified_review_count": 4,
        "product_price": 25.0,
        "product_review_count": 50,
    }
    for col, default_val in defaults.items():
        if col not in df.columns or df[col].isnull().all():
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    df_feat = pd.DataFrame(index=df.index)

    # Core rating & discrepancy signals
    df_feat["rating"] = df["rating"].astype(float)
    df_feat["product_avg_rating"] = df["product_avg_rating"].astype(float)
    df_feat["rating_discrepancy"] = df_feat["rating"] - df_feat["product_avg_rating"]
    df_feat["abs_rating_discrepancy"] = np.abs(df_feat["rating_discrepancy"])
    df_feat["reviewer_avg_rating"] = df["reviewer_avg_rating"].astype(float)
    df_feat["reviewer_rating_discrepancy"] = df_feat["rating"] - df_feat["reviewer_avg_rating"]

    # Transactional & behavioral signals
    df_feat["verified_purchase"] = df["verified_purchase"].astype(float)
    df_feat["reviewer_review_count"] = np.log1p(df["reviewer_review_count"].clip(lower=0).astype(float))
    df_feat["days_since_previous_review"] = np.log1p(df["days_since_previous_review"].clip(lower=0).astype(float))
    df_feat["reviewer_account_age_days"] = np.log1p(df["reviewer_account_age_days"].clip(lower=0).astype(float))

    # Reviewer reliability ratio
    rev_tot = df["reviewer_review_count"].astype(float)
    rev_ver = df["reviewer_verified_review_count"].astype(float)
    df_feat["reviewer_verified_ratio"] = (rev_ver / (rev_tot + 1e-5)).clip(0.0, 1.0)

    # Product popularity and price
    df_feat["product_price"] = np.log1p(df["product_price"].clip(lower=0).astype(float))
    df_feat["product_review_count"] = np.log1p(df["product_review_count"].clip(lower=0).astype(float))

    feature_cols = list(df_feat.columns)
    feat_matrix = df_feat[feature_cols].values.astype(np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

    if fit_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feat_matrix = scaler.fit_transform(feat_matrix)
        return feat_matrix, scaler, feature_cols
    else:
        if scaler is not None:
            feat_matrix = scaler.transform(feat_matrix)
        return feat_matrix, feature_cols


class DeBERTaTabularFusionNet(nn.Module):
    """
    Multimodal Deep Learning Architecture combining DeBERTa contextual text representations
    with engineered behavioral tabular metadata for fake review detection.
    """
    def __init__(self, text_dim=768, tab_dim=13, hidden_dim=64, dropout=0.25):
        super().__init__()
        # Text projection branch
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Tabular metadata branch
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        # Multimodal fusion layer
        fusion_dim = 128 + 32
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, text_embeds: torch.Tensor, tab_features: torch.Tensor) -> torch.Tensor:
        text_rep = self.text_mlp(text_embeds)
        tab_rep = self.tab_mlp(tab_features)
        fused = torch.cat([text_rep, tab_rep], dim=-1)
        logits = self.fusion_head(fused)
        return logits
