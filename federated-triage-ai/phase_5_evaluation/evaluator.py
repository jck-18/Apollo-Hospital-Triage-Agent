"""
Testing suite to objectively compare AI models against a shared Ground Truth.
"""
import pandas as pd
from typing import Dict, Any
from .metrics import calculate_metrics
from phase_2_local_models.predict_local import predict

class ModelEvaluator:
    """
    Standardizes the testing procedure. 
    It ensures that Hospital A's model, the Meta-Learner, and the Federated Model 
    are all tested against the EXACT same global patients, guaranteeing a fair fight.
    """
    def __init__(self, X_global: pd.DataFrame, y_global: pd.Series):
        self.X_global = X_global
        self.y_global = y_global
        
    def evaluate_model(self, model: Any, model_name: str) -> Dict[str, Any]:
        """
        Forces a model to predict the global datasets and scores its performance.
        """
        # Ask the model to diagnose all the global patients
        predictions = predict(model, self.X_global)
        
        # Calculate exactly how well it did against the true diagnoses
        metrics = calculate_metrics(self.y_global, predictions)
        
        return {
            "name": model_name,
            "metrics": metrics
        }
