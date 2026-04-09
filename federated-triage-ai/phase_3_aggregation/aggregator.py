"""
Provides a clean wrapper to apply ensemble logic over registered models.
"""
from phase_2_local_models.model_registry import ModelRegistry
from phase_2_local_models.predict_local import predict, predict_proba
from .ensemble_strategies import majority_voting, weighted_averaging
import pandas as pd
import numpy as np
from typing import List

class PredictionAggregator:
    """
    Coordinates the gathering of predictions from all registered silos.
    This acts as our "Central Node" for the Ensembling Strategy.
    """
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def ensemble_predict_majority(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Polls all hospital models for a Yes/No answer, outputs majority vote.
        """
        preds = []
        for hosp_name, model in self.registry.get_all_models().items():
            preds.append(predict(model, X_test))
            
        return majority_voting(preds)
        
    def ensemble_predict_weighted(self, X_test: pd.DataFrame, weights: List[float] = None) -> np.ndarray:
        """
        Polls all hospital models for their % confidence, outputs weighted average.
        """
        probs = []
        for hosp_name, model in self.registry.get_all_models().items():
            probs.append(predict_proba(model, X_test))
            
        return weighted_averaging(probs, weights)
