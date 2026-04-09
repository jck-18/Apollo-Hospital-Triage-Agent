"""
Data generation and loading utilities.
"""
import pandas as pd
import numpy as np
from . import schema

def generate_synthetic_triage_data(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset mimicking hospital emergency triage records.
    Features: Age, Temperature, Heart Rate, Cough, Shortness of Breath, Travel History.
    Target: Urgency Label (0 = Low, 1 = High).
    """
    np.random.seed(random_state)
    
    # 1. Generate numerical features
    age = np.random.normal(50, 15, n_samples).clip(1, 100)
    temperature = np.random.normal(37.5, 1.0, n_samples).clip(35.0, 42.0)
    heart_rate = np.random.normal(80, 15, n_samples).clip(40, 160)
    
    # 2. Generate categorical features (Bernoulli distribution)
    cough = np.random.binomial(1, 0.4, n_samples)
    sob = np.random.binomial(1, 0.2, n_samples)
    travel = np.random.binomial(1, 0.1, n_samples)
    
    # 3. Simulate a realistic Target Variable (Urgency)
    # A patient's urgency score is correlated with high fever, heart rate, shortness of breath, etc.
    risk_score = (
        (age / 50.0) * 0.5 + 
        (temperature > 38.0).astype(int) * 1.5 + 
        (heart_rate > 100).astype(int) * 1.0 + 
        cough * 0.5 + 
        sob * 2.0 + 
        travel * 1.0 + 
        np.random.normal(0, 0.5, n_samples) # Add Gaussian noise
    )
    
    # Binarize score to get a classification target
    # Adjust threshold so data is relatively balanced
    threshold = np.percentile(risk_score, 70) # Top 30% are high urgency
    urgency_label = (risk_score > threshold).astype(int)
    
    # 4. Construct DataFrame
    df = pd.DataFrame({
        schema.AGE: age,
        schema.TEMPERATURE: temperature,
        schema.HEART_RATE: heart_rate,
        schema.SYMPTOM_COUGH: cough,
        schema.SYMPTOM_SHORTNESS_OF_BREATH: sob,
        schema.TRAVEL_HISTORY: travel,
        schema.TARGET: urgency_label
    })
    
    return df

def load_data(n_samples: int = 15000) -> pd.DataFrame:
    """
    Loads dataset. For this POC, we generate it on the fly to avoid external dependencies.
    """
    return generate_synthetic_triage_data(n_samples=n_samples)
