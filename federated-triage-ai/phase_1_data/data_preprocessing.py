"""
Data preprocessing transformations.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from . import schema

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies scaling and transformations to the dataset.
    Returns: Features (X) and Target Labels (y).
    """
    # Defensive copy to avoid Pandas chained assignment warnings
    df_processed = df.copy()
    
    # Isolate targets
    y = df_processed[schema.TARGET]
    
    # 1. Scale continuous/numerical features using standardization
    scaler = StandardScaler()
    df_processed[schema.NUMERICAL_FEATURES] = scaler.fit_transform(df_processed[schema.NUMERICAL_FEATURES])
    
    # Categorical features are already binary in our synthetic script (0/1).
    # If using string categories, we would OneHotEncode them here.
    
    # 2. Extract final Features (X) DataFrame
    X = df_processed[schema.FEATURES]
    
    return X, y
