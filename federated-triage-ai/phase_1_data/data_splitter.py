"""
Logic to partition data into separate non-overlapping hospital datasets.
"""
import pandas as pd
from typing import Dict, Tuple

def split_into_hospitals(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_hospitals: int = 3, 
    random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Splits the central dataframe into N distinct slices representing hospital silos.
    This simulates horizontal data partitioning, where each hospital has entirely 
    isolated records.
    
    Returns:
    A dictionary mapping hospital names to a tuple of (X_train, y_train) data.
    """
    # Defensive copy
    X_shuffled = X.sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_shuffled = y.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    splits = dict()
    chunk_size = len(X_shuffled) // n_hospitals
    
    hospital_names = [f"hospital_{chr(65+i)}" for i in range(n_hospitals)] # hospital_A, hospital_B, etc.
    
    for i, name in enumerate(hospital_names):
        start_idx = i * chunk_size
        
        # The last hospital takes all remaining rows (to handle uneven division)
        end_idx = (i + 1) * chunk_size if i < n_hospitals - 1 else len(X_shuffled)
        
        X_chunk = X_shuffled.iloc[start_idx:end_idx].copy()
        y_chunk = y_shuffled.iloc[start_idx:end_idx].copy()
        
        splits[name] = (X_chunk, y_chunk)
        
    return splits
