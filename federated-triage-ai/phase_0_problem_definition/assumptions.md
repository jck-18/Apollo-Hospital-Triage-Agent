# System Assumptions and Constraints

## 1. Simulation Scope
- This is a localized architectural proof-of-concept (POC). No APIs, remote HTTP networking, sockets, or distributed cloud orchestration frameworks (e.g., Kubernetes) will be used.
- "Hospitals" (clients) and the "Central Server" function as logical units running natively in a single Python process memory space to simulate the Federated Learning loop.

## 2. Dataset Constraints
- **Data Privacy**: No patient records or raw telemetry parameters crossover between the horizontal partitions (Hospital datasets). The central aggregator is strictly denied visibility into raw features.
- **Synthesized Triage Data**: For simplicity, a standard public dataset (or generated mock data) representing features like Age, Vital Signs, Travel History, Symptoms, and Output Triage Urgency will be used.
- **IID vs Non-IID Distribution**: We assume the dataset splits are generally IID (Independent and Identically Distributed), though in real-world scenarios, datasets often suffer from local demographic skew (Non-IID). We will ensure the data is randomly distributed among the 3 clients to simulate a baseline environment without having to implement complex Non-IID compensation mechanisms (like FedProx) immediately.

## 3. Machine Learning Configuration
- Model architecture will be purposely simple to emphasize learning loops rather than state-of-the-art predictive accuracy. `scikit-learn`'s `SGDClassifier` or a minimalist PyTorch `nn.Sequential` multi-layer perceptron (MLP) will be used to simulate weight extractions and SGD-based learning.
- Synchronous updating: The server waits for all 3 clients to finish training their local epochs before aggregating weights for the current round (`R`).

## 4. Hardware and Scale
- CPU-only execution is acceptable. Small, computationally inexpensive models are utilized for fast turnaround in testing loops. No GPU (CUDA) is required.
- Memory consumption is nominal. Data fits reasonably inside the machine's RAM.
