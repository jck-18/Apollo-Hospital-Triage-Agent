"""
Training loop logic for specific hospital data.
"""
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from typing import Optional

def train_model(
    model: SGDClassifier, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    epochs: int = 5,
    classes: Optional[np.ndarray] = None
) -> SGDClassifier:
    """
    Trains a model locally using mini-batch Stochastic Gradient Descent logic.
    Instead of a single blind `.fit()`, we iterate over epochs with `.partial_fit()`.
    
    Args:
        model: Untrained or pre-trained SGD classifier instance.
        X_train: Local features dataframe.
        y_train: Local targets series.
        epochs: Number of complete passes over the data (Local Epochs 'E' in Federated Learning).
        classes: All possible target classes (e.g., [0, 1]).
        
    Returns:
        The trained SGDClassifier model.
    """
    if classes is None:
        classes = np.array([0, 1])  # Default to binary triage urgency
        
    # Standard warning: If a hospital doesn't have all classes natively in its dataset,
    # partial_fit requires knowing the global domain of classes beforehand.
    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=classes)
        
    return model
