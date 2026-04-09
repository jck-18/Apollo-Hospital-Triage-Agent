import pandas as pd
from sklearn.metrics import accuracy_score

from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

from phase_2_local_models.model_factory import create_local_model
from phase_2_local_models.train_local import train_model
from phase_2_local_models.predict_local import predict
from phase_2_local_models.model_registry import ModelRegistry

def main():
    print("--- Phase 2: Local Models Demo ---\n")
    
    # 1. Pipeline from Phase 1
    print("1. Generating scaled data and splitting across hospitals...")
    X, y = preprocess_data(load_data(n_samples=3000))
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    
    # 2. Initialize Model Registry
    registry = ModelRegistry()
    
    # 3. Train Individual Models inside their Silos
    print("\n2. Launching Local Model Training:")
    for hosp_name, (X_hosp, y_hosp) in hospitals.items():
        # Fake a train/test split within the hospital
        train_size = int(0.8 * len(X_hosp))
        X_train, y_train = X_hosp.iloc[:train_size], y_hosp.iloc[:train_size]
        X_test, y_test   = X_hosp.iloc[train_size:], y_hosp.iloc[train_size:]
        
        # Instantiate Model
        model = create_local_model(random_state=42)
        
        # Train
        trained_model = train_model(model, X_train, y_train, epochs=10)
        
        # Save to Registry
        registry.register_model(hosp_name, trained_model)
        
        # Local Prediction Evaluation
        preds = predict(trained_model, X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"   [+] {hosp_name} Model created. Local Test Accuracy: {acc * 100:.2f}%")
        
    print(f"\n3. Model Registry Content: {registry.list_models()}")

if __name__ == "__main__":
    main()
