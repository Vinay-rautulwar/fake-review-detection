"""
Module for extracting behavioral features from user and product interactions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class BehavioralFeatureExtractor:
    """
    Extracts behavioral features for users and products from review data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): DataFrame containing review_id, user_id, product_id, rating, text, date
        """
        self.df = df.copy()
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            
        if 'text' in self.df.columns:
            self.df['text_str'] = self.df['text'].astype(str)
            self.df['word_count'] = self.df['text_str'].apply(lambda x: len(x.split()))
            self.df['capital_ratio'] = self.df['text_str'].apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
            )
            
    def extract_user_features(self) -> pd.DataFrame:
        """Extracts per-user features."""
        prod_avg = self.df.groupby('product_id')['rating'].mean().rename('prod_avg_rating')
        df_merged = self.df.merge(prod_avg, on='product_id', how='left')
        df_merged['rating_dev'] = abs(df_merged['rating'] - df_merged['prod_avg_rating'])
        
        user_group = df_merged.groupby('user_id')
        
        user_features = pd.DataFrame(index=user_group.groups.keys())
        user_features.index.name = 'user_id'
        
        user_features['review_count'] = user_group.size()
        user_features['avg_rating'] = user_group['rating'].mean()
        user_features['rating_std'] = user_group['rating'].std().fillna(0)
        
        user_features['avg_review_length'] = user_group['word_count'].mean()
        user_features['review_length_std'] = user_group['word_count'].std().fillna(0)
        user_features['singleton_reviewer'] = (user_features['review_count'] == 1).astype(int)
        
        user_features['rating_deviation'] = user_group['rating_dev'].mean()
        user_features['capital_letter_ratio'] = user_group['capital_ratio'].mean()
        
        if 'date' in self.df.columns:
            first_review = user_group['date'].min()
            last_review = user_group['date'].max()
            user_features['first_last_span_days'] = (last_review - first_review).dt.days
            
            daily_counts = df_merged.groupby(['user_id', df_merged['date'].dt.date]).size()
            user_features['max_reviews_per_day'] = daily_counts.groupby('user_id').max()
        else:
            user_features['first_last_span_days'] = 0
            user_features['max_reviews_per_day'] = 0
            
        return user_features.reset_index()

    def _entropy(self, series: pd.Series) -> float:
        """Calculates Shannon entropy of a distribution."""
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-9))

    def extract_product_features(self) -> pd.DataFrame:
        """Extracts per-product features."""
        prod_group = self.df.groupby('product_id')
        
        prod_features = pd.DataFrame(index=prod_group.groups.keys())
        prod_features.index.name = 'product_id'
        
        prod_features['product_review_count'] = prod_group.size()
        prod_features['product_avg_rating'] = prod_group['rating'].mean()
        prod_features['product_rating_std'] = prod_group['rating'].std().fillna(0)
        
        prod_features['product_rating_entropy'] = prod_group['rating'].apply(self._entropy)
        
        prod_features['product_extreme_ratio'] = prod_group['rating'].apply(
            lambda x: sum((x == 1) | (x == 5)) / len(x)
        )
        
        return prod_features.reset_index()

    def extract_all_features(self) -> pd.DataFrame:
        """Merges user and product features back to the review-level dataframe."""
        user_features = self.extract_user_features()
        prod_features = self.extract_product_features()
        
        merged_df = self.df.copy()
        merged_df = merged_df.merge(user_features, on='user_id', how='left')
        merged_df = merged_df.merge(prod_features, on='product_id', how='left')
        
        cols_to_drop = ['text_str', 'word_count', 'capital_ratio']
        merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], inplace=True)
        
        return merged_df

    def normalize_features(self, features: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        """
        Normalizes specified feature columns using StandardScaler.
        
        Args:
            features: DataFrame containing the features.
            feature_cols: List of column names to normalize. If None, normalizes all numeric columns 
                          except id and label columns.
        """
        df_norm = features.copy()
        
        if feature_cols is None:
            exclude = ['review_id', 'user_id', 'product_id', 'label', 'text', 'cleaned_text', 'date']
            feature_cols = [c for c in df_norm.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_norm[c])]
            
        scaler = StandardScaler()
        if feature_cols:
            df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
            
        return df_norm
