"""
Utilities for extracting and mathematically averaging Artificial Intelligence parameter weights.
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from typing import List, Tuple

def get_weights(model: SGDClassifier) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the learned mathematical relationships (weights and bias) from the model.
    This is the ONLY data that leaves the hospital.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model has not been initialized with data yet.")
    return model.coef_.copy(), model.intercept_.copy()

def set_weights(model: SGDClassifier, coef: np.ndarray, intercept: np.ndarray) -> SGDClassifier:
    """
    Overwrites the local model's brain with the new Global Model's parameters.
    """
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()
    return model

def average_weights(weights_list: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Core Federated Averaging (FedAvg) Algorithm.
    It takes the weights from Hospital A, B, and C, and physically averages them.
    Because mathematical weights act as vectors in hyperspace, their average represents
    a model that has universally generalized the features seen across all silos.
    """
    n = len(weights_list)
    avg_coef = sum(w[0] for w in weights_list) / n
    avg_intercept = sum(w[1] for w in weights_list) / n
    return avg_coef, avg_intercept
