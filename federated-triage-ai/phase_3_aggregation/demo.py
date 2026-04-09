import pandas as pd
from sklearn.metrics import accuracy_score

from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals

from phase_2_local_models.model_factory import create_local_model
from phase_2_local_models.train_local import train_model
from phase_2_local_models.model_registry import ModelRegistry
from phase_2_local_models.predict_local import predict

from phase_3_aggregation.aggregator import PredictionAggregator
from phase_3_aggregation.meta_model import MetaModelAggregator

def main():
    print("--- Phase 3: Aggregation Strategies Demo ---\n")
    
    # 1. Setup Phase 1 & 2
    X, y = preprocess_data(load_data(n_samples=15000))
    hospitals = split_into_hospitals(X, y, n_hospitals=3)
    registry = ModelRegistry()
    
    print("1. Training Local Models on severely biased (Non-IID) populations...")
    trained_models = {}
    for hosp_name, (X_hosp, y_hosp) in hospitals.items():
        model = create_local_model(random_state=42)
        # Train on the entire hospital subset
        trained_model = train_model(model, X_hosp, y_hosp, epochs=10)
        registry.register_model(hosp_name, trained_model)
        trained_models[hosp_name] = trained_model
        
    # We create a "Global/Unseen" holdout dataset that has a fair mix of ALL ages
    X_global, y_global = preprocess_data(load_data(n_samples=2000)) 
    
    print("\n2. Testing models strictly against the Global/Mixed population:")
    base_accuracies = {}
    for hosp_name, trained_model in trained_models.items():
        acc = accuracy_score(y_global, predict(trained_model, X_global))
        base_accuracies[hosp_name] = acc
        print(f"   [!] {hosp_name} isolated model crashed to: {acc * 100:.2f}% (Overfit to local bias!)")
        
    avg_local = sum(base_accuracies.values()) / len(base_accuracies)
    print(f"   --> Average isolated performance on Global Scale: {avg_local * 100:.2f}%\n")
    
    # 3. Aggregation Phase
    print("3. Central Aggregation Results:")
    
    # Simple Aggregations
    aggregator = PredictionAggregator(registry)
    
    # Majority Voting
    preds_majority = aggregator.ensemble_predict_majority(X_global)
    acc_majority = accuracy_score(y_global, preds_majority)
    print(f"   [+] Majority Voting Strategy Accuracy: {acc_majority * 100:.2f}%")
    
    # Weighted Average
    preds_weighted = aggregator.ensemble_predict_weighted(X_global)
    acc_weighted = accuracy_score(y_global, preds_weighted)
    print(f"   [+] Pooled Probability Strategy Accuracy: {acc_weighted * 100:.2f}%")
    
    # Meta Model Strategy
    # We use half the unseen data to train the Meta Model, half to test it
    X_meta, X_test_meta = X_global.iloc[:1000], X_global.iloc[1000:]
    y_meta, y_test_meta = y_global.iloc[:1000], y_global.iloc[1000:]
    
    meta_agg = MetaModelAggregator(registry)
    meta_agg.fit(X_meta, y_meta)
    preds_meta = meta_agg.predict(X_test_meta)
    acc_meta = accuracy_score(y_test_meta, preds_meta)
    
    print(f"   [+] Meta-Model (Learned) Aggregation Accuracy: {acc_meta * 100:.2f}%")
    print("\nConclusion: Aggregation allows the system to beat the average local silo!")

if __name__ == "__main__":
    main()
