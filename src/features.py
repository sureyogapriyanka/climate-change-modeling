"""
features.py — handles feature engineering (dates, encoding, text)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-based features"""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    return df


def encode_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Encode profile names as numeric IDs"""
    encoder = LabelEncoder()
    df['profile_encoded'] = encoder.fit_transform(df['profileName'])
    return df


def build_feature_matrix(df: pd.DataFrame, target_col="likesCount", max_features=500):
    """Combine text (TF-IDF) and metadata into one feature matrix"""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_text = tfidf.fit_transform(df['text'])
    X_meta = df[['commentsCount', 'year', 'month', 'day', 'weekday', 'profile_encoded']]
    y = df[target_col]
    X_final = hstack([X_text, X_meta.values])
    return X_final, y, tfidf
