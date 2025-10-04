"""
data.py — handles data loading, saving, and cleaning
"""

import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset"""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset from {path} — shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic cleanup"""
    df['text'] = df['text'].fillna("")
    df['profileName'] = df['profileName'].fillna("Unknown")
    df['likesCount'] = df['likesCount'].fillna(0)
    df['commentsCount'] = df['commentsCount'].fillna(0)
    return df


def save_processed(df: pd.DataFrame, out_path: str):
    """Save cleaned data"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"💾 Processed data saved to: {out_path}")
