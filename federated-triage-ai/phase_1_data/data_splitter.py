"""
Logic to partition data into separate non-overlapping hospital datasets.
"""
import pandas as pd
from typing import Dict, Tuple

from . import schema

def split_into_hospitals(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_hospitals: int = 3, 
    random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Splits the central dataframe into N distinct slices representing hospital silos.
    We introduce a realistic Non-IID (80/10/10) skew here by sorting patients 
    by Age, dividing them into young/mid/old pools, and giving each hospital 
    80% of its dominant demographic, and 10% of the other two demographics.
    """
    df_combined = X.copy()
    df_combined['__temp_target'] = y
    
    # 1. Sort by Demographic (Age)
    df_combined = df_combined.sort_values(by=schema.AGE).reset_index(drop=True)
    
    # 2. Chop into exact thirds (Young, Middle, Old pools)
    thirds = len(df_combined) // 3
    pools = {
        "young": df_combined.iloc[:thirds].sample(frac=1, random_state=random_state),
        "mid": df_combined.iloc[thirds:2*thirds].sample(frac=1, random_state=random_state),
        "old": df_combined.iloc[2*thirds:].sample(frac=1, random_state=random_state)
    }
    
    splits = {"hospital_A": [], "hospital_B": [], "hospital_C": []}
    
    def delegate_pool(pool, dominant_hosp, hosp1, hosp2, dominant_ratio=0.8):
        """Disperses a specific pool of patients 80% to one hospital, 10% to others."""
        n = len(pool)
        dom_idx = int(n * dominant_ratio)
        half_rem = int(n * ((1.0 - dominant_ratio) / 2))
        
        splits[dominant_hosp].append(pool.iloc[:dom_idx])
        splits[hosp1].append(pool.iloc[dom_idx:dom_idx + half_rem])
        splits[hosp2].append(pool.iloc[dom_idx + half_rem:])
        
    # Hospital A dominates the young pool, B the mid pool, C the old pool
    delegate_pool(pools["young"], "hospital_A", "hospital_B", "hospital_C")
    delegate_pool(pools["mid"],   "hospital_B", "hospital_A", "hospital_C")
    delegate_pool(pools["old"],   "hospital_C", "hospital_A", "hospital_B")
    
    final_splits = dict()
    
    # 3. Rebuild the dataset for each hospital and shuffle them thoroughly
    for name, blocks in splits.items():
        chunk = pd.concat(blocks)
        chunk = chunk.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        y_chunk = chunk.pop('__temp_target')
        final_splits[name] = (chunk, y_chunk)
        
    return final_splits
