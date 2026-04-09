"""
Factory logic for creating standardized AI models for our hospitals.
"""
from sklearn.linear_model import SGDClassifier

def create_local_model(random_state: int = 42) -> SGDClassifier:
    """
    Creates a fresh, untrained local model.
    
    Why SGDClassifier?
    By using SGD (Stochastic Gradient Descent) with a log_loss objective, 
    we are effectively creating a single-layer neural network (Logistic Regression).
    This is highly advantageous for our Federated Learning POC because:
    1. We can train in multiple epochs using `partial_fit()`.
    2. We can easily extract and average its mathematical weights (`coef_` and `intercept_`) in Phase 4 
       without needing a massive framework like PyTorch.
       
    Returns:
        Untrained SGDClassifier instance.
    """
    return SGDClassifier(
        loss='log_loss', 
        learning_rate='constant', 
        eta0=0.01, # Standard learning rate
        max_iter=1,  # We control epochs manually
        warm_start=True, # Remember weights between partial_fit calls
        random_state=random_state
    )
