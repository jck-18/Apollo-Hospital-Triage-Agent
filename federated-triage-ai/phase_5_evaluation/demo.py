import pandas as pd

from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

from phase_4_federated_learning.client_node import HospitalClientNode
from phase_4_federated_learning.server_aggregator import FederatedServer
from phase_4_federated_learning.federated_trainer import FederatedPipelineRunner

from phase_3_aggregation.aggregator import PredictionAggregator
from phase_3_aggregation.meta_model import MetaModelAggregator

from phase_5_evaluation.evaluator import ModelEvaluator
from phase_5_evaluation.comparison import print_comparison_table
from phase_2_local_models.model_registry import ModelRegistry

def main():
    print("--- Phase 5: Comprehensive Evaluation Demo ---\n")
    
    # 1. Setup the 80/10/10 Non-IID Environment
    print("1. Generating 80/10/10 Non-IID Hospital Data...")
    X, y = preprocess_data(load_data(n_samples=15000))
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    
    # 2. Setup Federated Environment & Model Registry
    clients = []
    registry = ModelRegistry()
    for hosp_name, (X_hosp, y_hosp) in hospitals.items():
        client = HospitalClientNode(hosp_name, X_hosp, y_hosp)
        clients.append(client)
        # Register the client's internal model to the registry so Phase 3 tools can track them
        registry.register_model(hosp_name, client.model)
        
    server = FederatedServer()
    trainer = FederatedPipelineRunner(clients, server)
    
    # 3. Train all baseline systems
    print("\n2. Executing Federated Training Loop (5 Rounds)...")
    global_model = trainer.run_training_loop(rounds=5, local_epochs=3)
    
    # Re-register the fully trained localized models into the registry for Phase 3 aggregation
    for c in clients:
        registry.register_model(c.hospital_id, c.model)
        
    print("\n3. Booting Phase 3 Aggregators...")
    simple_agg = PredictionAggregator(registry)
    
    # 4. Generate the Grand Unified Holdout Dataset
    print("\n4. Generating 2,000 Unseen Global Patients for the Ultimate Test...")
    X_global, y_global = preprocess_data(load_data(n_samples=2000))
    evaluator = ModelEvaluator(X_global, y_global)
    
    # 5. Run Evaluations
    results = []
    
    # Evaluate baselines (Isolated Silos)
    for c in clients:
        results.append(evaluator.evaluate_model(c.model, f"Isolated Baseline: {c.hospital_id}"))
        
    # Phase 3 requires creating dummy 'Predict wrapper' functions to match the Evaluator interface
    # but we can just map it easily using a wrapper class
    class EnsembleWrapper:
        def __init__(self, func): self.func = func
        def predict(self, X): return self.func(X)
        
    # We patch the predict package requirement mechanically for Phase 3
    import phase_2_local_models.predict_local
    original_predict = phase_2_local_models.predict_local.predict
    def flexible_predict(model, X):
        if hasattr(model, 'func'): return model.func(X)
        return original_predict(model, X)
    phase_2_local_models.predict_local.predict = flexible_predict
    
    results.append(evaluator.evaluate_model(EnsembleWrapper(simple_agg.ensemble_predict_majority), "Phase 3: Majority Vote"))
    results.append(evaluator.evaluate_model(EnsembleWrapper(simple_agg.ensemble_predict_weighted), "Phase 3: Soft Vote"))
    
    results.append(evaluator.evaluate_model(global_model, "Phase 4: Federated True Model"))
    
    # 6. Render Beautiful Report
    print_comparison_table(results)

if __name__ == "__main__":
    main()
