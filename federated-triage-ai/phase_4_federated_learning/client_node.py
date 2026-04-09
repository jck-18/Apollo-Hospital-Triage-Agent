"""
Represents a physical hospital compute node in the Federated Learning network.
"""
import pandas as pd
from phase_2_local_models.train_local import train_model
from phase_2_local_models.model_factory import create_local_model
from .weight_utils import get_weights, set_weights
from typing import Tuple
import numpy as np

class HospitalClientNode:
    """
    A simulated secure hospital environment that never allows its internal dataset 
    to be accessed from the outside.
    """
    def __init__(self, hospital_id: str, X_train: pd.DataFrame, y_train: pd.Series):
        self.hospital_id = hospital_id
        
        # Privacy Constraint: Data is stored entirely within the client runtime boundary
        self._X_train = X_train
        self._y_train = y_train
        
        # Initialize the baseline model structures
        self.model = create_local_model()
        
        # We run a tiny dummy `.partial_fit` to force scikit-learn to initialize the 
        # `model.coef_` arrays so they exist in memory before we start federating.
        self.model.partial_fit(self._X_train.iloc[:2], self._y_train.iloc[:2], classes=np.array([0, 1]))
        
    def train_on_global_weights(self, global_coef: np.ndarray, global_intercept: np.ndarray, epochs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Federated Learning Step:
        1. Download Global Brain (parameters) from Server.
        2. Inject them into local model.
        3. Train for `E` Epochs locally on isolated patient data.
        4. Extract the updated weights and return them to the Server.
        """
        if global_coef is not None and global_intercept is not None:
            self.model = set_weights(self.model, global_coef, global_intercept)
            
        self.model = train_model(self.model, self._X_train, self._y_train, epochs=epochs)
        
        return get_weights(self.model)
