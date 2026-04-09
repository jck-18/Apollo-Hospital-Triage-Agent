"""
Schema definitions for the patient triage dataset.
"""

# Feature columns
AGE = "age"
TEMPERATURE = "temperature"  # in Celsius
HEART_RATE = "heart_rate"    # beats per minute
SYMPTOM_COUGH = "symptom_cough"  # Binary 0 or 1
SYMPTOM_SHORTNESS_OF_BREATH = "symptom_shortness_of_breath"  # Binary 0 or 1
TRAVEL_HISTORY = "travel_history"  # Binary 0 or 1

NUMERICAL_FEATURES = [AGE, TEMPERATURE, HEART_RATE]
CATEGORICAL_FEATURES = [SYMPTOM_COUGH, SYMPTOM_SHORTNESS_OF_BREATH, TRAVEL_HISTORY]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Target column (0 = Low Urgency, 1 = High/Critical Urgency)
TARGET = "urgency_label"
