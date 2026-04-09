"""
Inference logic for local testing.
"""
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

def predict(model: SGDClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Generates deterministic binary predictions (0 or 1) using a 
    locally trained model for triage urgency.
    
    Args:
        model: Trained SGDClassifier.
        X_test: Input features dataset.
        
    Returns:
        1D numpy array of integer class labels.
    """
    return model.predict(X_test)
    
def predict_proba(model: SGDClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Calculates the raw probability scores of belonging to the Urgency class.
     Useful for soft-voting ensemble aggregations natively in Phase 3.
     
    Args:
        model: Trained SGDClassifier.
        X_test: Input features dataset.
        
    Returns:
        2D numpy array of shape (n_samples, 2), where column 0 is 
        Probability of Class 0 (Low), and col 1 is Prob of Class 1 (High).
    """
    return model.predict_proba(X_test)
