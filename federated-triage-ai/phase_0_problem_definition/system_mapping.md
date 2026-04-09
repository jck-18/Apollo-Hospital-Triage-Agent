# System Architecture & Mapping

## Components

The system consists of the following modular layers, each reflecting a specific phase of the federated learning pipeline:

### 1. Data Layer (`phase_1_data/`)
- Responsible for parsing raw demographic and clinical data.
- Handles feature engineering and normalization.
- **Data Splitting**: Fragments the dataset horizontally across 3 hospital "silos", ensuring no data overlaps locally.

### 2. Local Models Layer (`phase_2_local_models/`)
- A factory method instantiates a uniform base model (e.g., Logistic Regression or Neural Network) for triage prediction.
- Each hospital maintains a private local instance of the model.
- Includes isolated training routines allowing hospitals to train only on their data splits.

### 3. Aggregation Layer (`phase_3_aggregation/`)
- Focuses on prediction-level aggregation. 
- Serves as the baseline strategy comparing raw ensemble techniques (Majority Voting, Weighted Averaging) against individual performances.

### 4. Federated Learning Layer (`phase_4_federated_learning/`)
- Implements the core **Federated Averaging (FedAvg)** algorithm loop:
    1. **Server** broadcasts global model parameters.
    2. **Clients** (hospitals) perform $E$ epochs of local training.
    3. **Clients** transmit updated parameters back to the Server.
    4. **Server** aggregates parameter weights securely to update the global model.

### 5. Evaluation Layer (`phase_5_evaluation/`)
- Standardized metric tracking across local silos and global models.
- Logs Accuracy, F1-Scores, Precision, and Recall for uniform benchmarking.

### 6. Visualization & Tooling (`phase_6_visualization/` and `phase_7_tooling/`)
- Visualization of real-time metrics, confusion matrices, and the comparison of global convergence vs point-in-time local models.
- Command-line interfaces and experiment runners to execute the simulation dynamically.

## Component Interaction Flow

1. **Initialization:** CLI initiates the Data Splitter.
2. **Setup:** The Model Registry spawns clients representing Hospital A, Hospital B, and Hospital C.
3. **Local Training:** Each hospital builds a localized point-of-view model.
4. **Federated Orchestration:** The Server cycles communication rounds, polling clients for weights and disseminating aggregated updates.
5. **Report Generation:** Finally, the Evaluator contrasts models visually via the CLI Pipeline Runner.
