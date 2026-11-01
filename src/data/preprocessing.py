"""
Module for text preprocessing.
"""

import re
import pandas as pd
from typing import Dict, Any, Union

def clean_html(text: str) -> str:
    """Removes HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_urls(text: str) -> str:
    """Removes URLs from text."""
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def normalize_text(text: str) -> str:
    """Normalizes whitespace and converts to lowercase."""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextPreprocessor:
    """Handles cleaning and tokenizing text."""
    
    def __init__(self):
        pass
        
    def clean_text(self, text: str) -> str:
        """Applies a sequence of cleaning steps to the text."""
        text = str(text)
        text = clean_html(text)
        text = remove_urls(text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"]', '', text)
        text = normalize_text(text)
        return text

    def tokenize(self, text: str, tokenizer: Any, max_length: int) -> Dict[str, Union[list, 'torch.Tensor']]:
        """
        Tokenizes the text using a HuggingFace tokenizer.
        """
        return tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

def preprocess_dataset(df: pd.DataFrame, tokenizer: Any, max_length: int) -> pd.DataFrame:
    """
    Preprocesses the text column of the entire dataframe.
    
    Args:
        df: DataFrame containing a 'text' column.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text and optionally tokenized data.
    """
    preprocessor = TextPreprocessor()
    df_clean = df.copy()
    
    if 'text' in df_clean.columns:
        df_clean['cleaned_text'] = df_clean['text'].apply(preprocessor.clean_text)
        
    return df_clean
