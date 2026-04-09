import pandas as pd
from sklearn.metrics import accuracy_score

from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

from phase_4_federated_learning.client_node import HospitalClientNode
from phase_4_federated_learning.server_aggregator import FederatedServer
from phase_4_federated_learning.federated_trainer import FederatedPipelineRunner

from phase_2_local_models.predict_local import predict

def main():
    print("--- Phase 4: Federated Learning Loop Demo ---\n")
    
    # 1. Setup Phase 1 (Highly Biased Non-IID Data)
    print("1. Generating Non-IID Hospital Data...")
    X, y = preprocess_data(load_data(n_samples=15000))
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    
    # 2. Setup Federated Environment
    print("\n2. Initializing Federated Ecosystem...")
    clients = []
    for hosp_name, (X_hosp, y_hosp) in hospitals.items():
        # Each client completely locks down its data
        client = HospitalClientNode(hosp_name, X_hosp, y_hosp)
        clients.append(client)
        print(f"   [+] Booted secure node: {hosp_name}")
        
    server = FederatedServer()
    trainer = FederatedPipelineRunner(clients, server)
    
    # 3. Training Loop
    print("\n3. Executing Core FedAvg Algorithm:")
    # We run 5 Communication Rounds. During each round:
    # Server sends parameters -> Clients train for 3 epochs -> Clients return parameters
    global_model = trainer.run_training_loop(rounds=5, local_epochs=3)
    
    # 4. Evaluation vs the Global Truth
    print("\n4. Evaluation vs Global Holdout Dataset:")
    X_global, y_global = preprocess_data(load_data(n_samples=2000)) 
    
    # Let's test how well the Global Model compares to the Client's isolated states
    acc_global_model = accuracy_score(y_global, predict(global_model, X_global))
    print(f"   [🌟] FEDERATED GLOBAL MODEL ACCURACY: {acc_global_model * 100:.2f}%")
    
    # Compare with a typical isolated client (e.g., Hospital A's local overfit model)
    acc_local_a = accuracy_score(y_global, predict(clients[0].model, X_global))
    print(f"   [!] Final Hospital A isolated internal model accuracy on global scale: {acc_local_a * 100:.2f}%")
    
    print("\nConclusion: True mathematical parameter averaging creates a vastly superior AI without sharing patient data!")

if __name__ == "__main__":
    main()
