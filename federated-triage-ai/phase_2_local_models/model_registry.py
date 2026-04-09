"""
Central repository to store initialized models in memory.
"""
from typing import Dict, List, Any

class ModelRegistry:
    """
    Acts as a simulated dictionary/cache holding all local models
    trained by the independent hospitals in the Federated network.
    """
    def __init__(self):
        self._models: Dict[str, Any] = {}
        
    def register_model(self, hospital_id: str, model: Any) -> None:
        """
        Saves a trained model instance to the registry mapped to its hospital.
        """
        self._models[hospital_id] = model
        
    def get_model(self, hospital_id: str) -> Any:
        """
        Retrieves a saved hospital model.
        Raises ValueError if hospital ID not found.
        """
        if hospital_id not in self._models:
            raise ValueError(f"Model for '{hospital_id}' not found in registry.")
        return self._models[hospital_id]
        
    def list_models(self) -> List[str]:
        """
        Returns a list of all registered hospital identifiers.
        """
        return list(self._models.keys())
        
    def get_all_models(self) -> Dict[str, Any]:
        """
        Returns a dictionary of all hospital IDs and their models.
        """
        return self._models.copy()
