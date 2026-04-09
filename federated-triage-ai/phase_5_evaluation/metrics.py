"""
Mathematical definitions for evaluating Artificial Intelligence model performance.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
import numpy as np

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Given the Ground Truth labels and the AI Predictions, calculates the 
    core 4 performance metrics used in medical diagnostic AI.
    
    Why these 4?
    - Accuracy: Overall correctness.
    - Precision: "When the AI says High Urgency, how often is it actually High Urgency?" (Prevents alarm fatigue).
    - Recall (Sensitivity): "Out of all the TRUE High Urgency cases, how many did the AI catch?" (Crucial so we don't send dying patients home).
    - F1 Score: The harmonic mean of Precision and Recall.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0)
    }
