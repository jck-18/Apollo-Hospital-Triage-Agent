# Federated Learning for Hospital Triage AI
### Proof-of-Concept: Progress Report & Proposal

> [!NOTE]
> **To the Reviewer:** This document outlines the rationale, architecture, and currently completed foundational phases of our Federated AI proof-of-concept. It serves as a checkpoint to review the system's strict privacy-preserving design before we integrate the core Federated Learning algorithms.

---

## 1. The Real-World Problem
Hospitals process vast amounts of patient data daily, particularly in Emergency Department triage. Training a highly accurate Artificial Intelligence model to predict a patient's **urgency level** requires massive, diverse datasets. 

However, **patient privacy laws (like HIPAA)** prevent hospitals from pooling raw patient data into a central database. Consequently:
- Small hospitals suffer from insufficient data, creating weak predictive models.
- Large hospitals have biased models that might not generalize to other regions or demographics.

## 2. Our Proposed Solution: Federated Learning
Instead of bringing the **data to the model**, we will bring the **model to the data**.

We are building a scalable Python-based simulation of a 3-hospital network. In this framework:
1. Each hospital securely trains a localized AI model on its own private patient data.
2. The hospitals extract only the **learned mathematical patterns (model weights/parameters)** and send them to a Central Server.
3. The Server aggregates these numbers to update a "Global Brain" without ever seeing a single raw patient record. The smarter, globally aware model is then sent back to the hospitals.

---

## 3. Work Completed So Far

We have laid down the exact foundations necessary to prove this concept works. The architecture is strictly compartmentalized to ensure modularity and data privacy.

### ✅ Phase 0 & 1: The Data Layer (Privacy Simulation)
We built a dynamic data simulator that generates thousands of realistic triage patient records, featuring:
- **Demographics & Vitals:** Age, Temperature, Heart Rate
- **Categorical Risk Factors:** Cough, Shortness of Breath, Travel History
- **Target Variable:** Urgency Level (Binary classification: High vs. Low Priority)

**The Privacy Constraint Simulated:**
To simulate the real world, the central dataset is immediately shattered horizontally into three isolated "silos" (`Hospital A`, `Hospital B`, and `Hospital C`). Data simply cannot leak between them.
```text
3. Splitting into Hospital Silos...
   ► hospital_A: Features (1000, 6), Labels (1000,)
   ► hospital_B: Features (1000, 6), Labels (1000,)
   ► hospital_C: Features (1000, 6), Labels (1000,)
```

### ✅ Phase 2: Local AI Models (Siloed Learning)
We constructed the Machine Learning factory using a lightweight algorithm (Logistic Regression trained via Stochastic Gradient Descent). We use this specific algorithm because it mirrors how Deep Neural Networks learn (using iterative Epochs) but is incredibly efficient, allowing us to easily extract weights later.

1. Each hospital receives a blank AI model.
2. The model trains exclusively on the hospital's isolated 1,000 patient records.
3. The models are stored safely in an in-memory `ModelRegistry`.

**Current Constraint (Why we need the next phase):**
Because these models only see a small, randomized sub-section of the population, their accuracy fluctuates significantly based on their local data bias.
```text
2. Launching Local Model Training:
   [+] hospital_A Model created. Local Test Accuracy: 81.50%
   [+] hospital_B Model created. Local Test Accuracy: 85.50%
   [+] hospital_C Model created. Local Test Accuracy: 87.00%
```

---

## 4. Proposed Next Steps (Pending Approval)

With these foundations built, we are ready to implement the Aggregation and Federated Learning algorithms.

- **Phase 3 — Aggregation:** We will first attempt to combine these 3 separate models using basic Ensembling (Majority Voting and Weighted Averaging). This will prove that 3 models working together beat 1 working alone.
- **Phase 4 — Federated Learning (FedAvg):** We will build the Central Server loop. The server will pull the weights from Hospital A, B, and C, mathematically average them, and overwrite the local models with the smarter global model.
- **Phase 5 & 6 — Evaluation & Visualization:** We will generate charts and confusion matrices proving that the new Federated Model achieves higher, more stable accuracy than any individual hospital could achieve on its own.

**Approval Request:** 
Everything we have built so far validates the pipeline's modularity and safety constraints. We request your review and approval to proceed to building the Core Aggregation and Federated Learning logic.
