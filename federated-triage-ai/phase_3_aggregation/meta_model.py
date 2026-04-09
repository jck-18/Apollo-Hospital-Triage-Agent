"""
Optional Stacking/Meta-Model Aggregation approach.
"""
from sklearn.linear_model import LogisticRegression
from phase_2_local_models.model_registry import ModelRegistry
from phase_2_local_models.predict_local import predict_proba
import pandas as pd
import numpy as np

class MetaModelAggregator:
    """
    A Stacking ensemble technique. 
    Instead of arbitrarily voting, we train a 'meta-learner' (a second-stage AI) 
    that looks at the probabilities outputted by Hospital A, B, and C, and physically 
    learns which hospital to "trust" the most for certain edge cases.
    """
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.meta_model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def _extract_base_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Runs the data through the Local Models. Their 'Confidence of High Urgency' 
        scores become the new "Features" for the Meta-Model.
        """
        features = []
        # Sort to ensure consistent mapping (A, B, C)
        for hosp_id in sorted(self.registry.list_models()):
            model = self.registry.get_model(hosp_id)
            # We only extract the probability for the "High Urgency" class (column 1)
            prob_high_urgency = predict_proba(model, X)[:, 1]
            features.append(prob_high_urgency)
            
        # Shape: (n_samples, n_hospitals)
        return np.column_stack(features) 
        
    def fit(self, X_meta: pd.DataFrame, y_meta: pd.Series) -> None:
        """
        Trains the meta-model using a holdout dataset so it can learn 
        how to optimally weigh the hospital opinions.
        """
        meta_features = self._extract_base_features(X_meta)
        self.meta_model.fit(meta_features, y_meta)
        self.is_trained = True
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generates the final 'smart' prediction."""
        if not self.is_trained:
            raise ValueError("Meta-model must be .fit() before predicting.")
            
        meta_features = self._extract_base_features(X_test)
        return self.meta_model.predict(meta_features)
