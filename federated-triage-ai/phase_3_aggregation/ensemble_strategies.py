"""
Mathematical strategies for combining the outputs of multiple AI models.
"""
import numpy as np
from typing import List

def majority_voting(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Takes a list of binary prediction arrays from different hospitals 
    and applies simple "majority rules" voting.
    
    Args:
        predictions_list: List of 1D numpy arrays containing 0s or 1s.
    Returns:
        1D numpy array with the majority decision.
    """
    # Stack into a 2D matrix where rows = models, cols = patients
    stacked = np.vstack(predictions_list) 
    
    # Tally up the "1" votes for each patient (down the columns)
    summed = np.sum(stacked, axis=0)
    
    # If strictly more than half the models predicted High Urgency (1)
    threshold = len(predictions_list) / 2.0
    ensemble_predictions = (summed > threshold).astype(int)
    
    return ensemble_predictions

def weighted_averaging(probabilities_list: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """
    Combines the raw probability scores of models using weighted averaging.
    This allows a more "confident" model to sway the final decision.
    
    Args:
        probabilities_list: List of 2D numpy arrays of shape (n_samples, 2).
        weights: Optional list of floats denoting importance. 
                 If None, equal weighting is applied.
    Returns:
        1D numpy array with the final binary decision.
    """
    n_models = len(probabilities_list)
    if weights is None:
        weights = [1.0 / n_models] * n_models
        
    # Stack into a 3D matrix (n_models, n_samples, 2_classes)
    stacked_probs = np.array(probabilities_list)
    
    # Calculate the weighted average across axis 0 (the different models)
    weighted_probs = np.average(stacked_probs, axis=0, weights=weights)
    
    # Class 1 (High Urgency) is located at column index 1
    # If combined average probability > 50%, predict 1
    ensemble_predictions = (weighted_probs[:, 1] > 0.5).astype(int)
    
    return ensemble_predictions
