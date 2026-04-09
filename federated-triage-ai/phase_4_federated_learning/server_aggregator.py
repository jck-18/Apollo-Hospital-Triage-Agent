"""
Represents the Central Server orchestration logic.
"""
from .weight_utils import average_weights
from typing import List, Tuple
import numpy as np

class FederatedServer:
    """
    The orchestrator of the ecosystem. 
    Notice that this server NEVER touches `X_train` or `y_train` from any hospital.
    It purely handles mathematics.
    """
    def __init__(self):
        self.global_coef = None
        self.global_intercept = None
        
    def aggregate_and_update(self, client_weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Takes the mathematical parameters from all localized hospitals and 
        generates the new Global Master Model.
        """
        self.global_coef, self.global_intercept = average_weights(client_weights)
        
    def get_global_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.global_coef, self.global_intercept
