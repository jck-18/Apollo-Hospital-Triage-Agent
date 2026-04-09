import os
import pandas as pd
from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

def export_to_csv():
    # Create datasets folder in the root project directory
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 15,000 global records (dynamic generation)...")
    df_raw = load_data()
    
    # Save the global un-split, un-preprocessed data
    global_path = os.path.join(output_dir, "global_triage_data.csv")
    df_raw.to_csv(global_path, index=False)
    print(f"[+] Global dataset saved to: {global_path} | Shape: {df_raw.shape}")
    
    # Preprocess and split into 3 exact silos
    X, y = preprocess_data(df_raw)
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    
    print("\nSplitting into Hospital Silos (5000 records each)...")
    for hosp_name, (X_hosp, y_hosp) in hospitals.items():
        # Combine X and y back together just for the CSV visual consistency
        df_hosp = X_hosp.copy()
        df_hosp['urgency_label'] = y_hosp
        
        file_path = os.path.join(output_dir, f"{hosp_name}_data.csv")
        df_hosp.to_csv(file_path, index=False)
        print(f"[+] Saved {hosp_name} data to: {file_path} | Shape: {df_hosp.shape}")
        
    print("\nAll data exported successfully! Your teacher can view these in Excel.")

if __name__ == "__main__":
    export_to_csv()
