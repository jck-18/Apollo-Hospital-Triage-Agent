"""
Data generation and loading utilities.
"""
import pandas as pd
import numpy as np
from . import schema

def generate_synthetic_triage_data(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset mimicking hospital emergency triage records.
    Features are statistically correlated with Age to reflect real-world demographics.
    """
    np.random.seed(random_state)
    
    # 1. Base Anchor Feature: Age
    age = np.random.normal(50, 15, n_samples).clip(1, 100)
    
    # 2. Correlated Numerical Features
    # In real life, younger patients (children) have higher resting heart rates 
    # than adults. Normal adult HR is ~60-100. Kid HR is ~80-120.
    heart_rate_base = 100 - (age * 0.3)
    heart_rate = np.random.normal(heart_rate_base, 15).clip(40, 170)
    
    # Core temperature tends to be very slightly lower in elderly populations
    temp_base = 37.5 - (age / 100) * 0.2
    temperature = np.random.normal(temp_base, 1.0).clip(35.0, 42.0)
    
    # 3. Correlated Categorical Features (Symptoms/History)
    # Older patients inherently have a higher risk of shortness of breath
    sob_probability = np.clip(0.05 + (age / 100.0) * 0.4, 0.0, 1.0)
    sob = np.random.binomial(1, sob_probability)
    
    # Younger/middle-aged patients generally travel more than the deeply elderly
    travel_probability = np.clip(0.4 - (age / 100.0) * 0.3, 0.0, 1.0)
    travel = np.random.binomial(1, travel_probability)
    
    # Cough is treated as completely independent (viral spread)
    cough = np.random.binomial(1, 0.4, n_samples)
    
    # 4. Simulate the Target Variable (Urgency)
    # Urgency naturally skews heavily with abnormal vitals and age
    risk_score = (
        (age / 50.0) * 0.7 + 
        (temperature > 38.0).astype(int) * 1.5 + 
        (heart_rate > 100).astype(int) * 1.0 + 
        cough * 0.5 + 
        sob * 2.0 + 
        travel * 1.0 + 
        np.random.normal(0, 0.5, n_samples) # Add Gaussian noise
    )
    
    # Binarize score to get a classification target
    threshold = np.percentile(risk_score, 70) # Top 30% are high urgency
    urgency_label = (risk_score > threshold).astype(int)
    
    # 5. Construct DataFrame
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
