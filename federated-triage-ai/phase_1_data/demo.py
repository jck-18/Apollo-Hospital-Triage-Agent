import pandas as pd
from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

def main():
    print("--- Phase 1: Data Layer Demo ---\n")
    
    # 1. Load Data
    print("1. Loading raw synthetic dataset...")
    df_raw = load_data(n_samples=3000)
    print(f"   Raw Dataset Shape: {df_raw.shape}")
    print(df_raw.head(3), "\n")
    
    # 2. Preprocess Data
    print("2. Preprocessing Data (Scaling Numerical Features)...")
    X, y = preprocess_data(df_raw)
    print(f"   Features Shape: {X.shape}, Target Shape: {y.shape}")
    print("   Sample Preprocessed Features:")
    print(X.head(3), "\n")
    
    # 3. Split Data
    print("3. Splitting into Hospital Silos...")
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    
    for name, (X_hosp, y_hosp) in hospitals.items():
        print(f"   ► {name}: Features {X_hosp.shape}, Labels {y_hosp.shape}")
        
if __name__ == "__main__":
    main()
