"""
Apollo HQ Triage Intelligence — FastAPI Backend
Wraps the Phase 1-5 Federated Learning engine into a web API.
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Ensure we can import all phase modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import google.generativeai as genai

from phase_1_data.data_loader import load_data
from phase_1_data.data_preprocessing import preprocess_data
from phase_1_data.data_splitter import split_into_hospitals
from phase_2_local_models.model_factory import create_local_model
from phase_2_local_models.train_local import train_model
from phase_2_local_models.model_registry import ModelRegistry
from phase_2_local_models.predict_local import predict, predict_proba
from phase_4_federated_learning.client_node import HospitalClientNode
from phase_4_federated_learning.server_aggregator import FederatedServer
from phase_4_federated_learning.federated_trainer import FederatedPipelineRunner
from phase_4_federated_learning.weight_utils import set_weights
from phase_5_evaluation.metrics import calculate_metrics

# ── Gemini Config ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE: Boot the entire ML pipeline once when the server starts
# ─────────────────────────────────────────────────────────────────────────────
print("🏥 [BOOT] Initializing Apollo HQ Federated ML Pipeline...")

X_full, y_full   = preprocess_data(load_data(n_samples=15000))
hospitals_data   = split_into_hospitals(X_full, y_full, n_hospitals=3)

# Global holdout for evaluation
X_eval, y_eval   = preprocess_data(load_data(n_samples=2000))

HOSPITAL_META = {
    "hospital_A": {"name": "Branch A — Metro Pediatric",  "dominant": "Pediatric / Youth",    "color": "#38bdf8"},
    "hospital_B": {"name": "Branch B — General Medicine", "dominant": "General Adult Care",    "color": "#a78bfa"},
    "hospital_C": {"name": "Branch C — Senior Care Hub",  "dominant": "Geriatric / Elderly",   "color": "#34d399"},
}

FEATURE_NAMES = ["Age", "Temperature", "Heart Rate", "Cough", "Shortness of Breath", "Travel History"]

clients: list[HospitalClientNode] = []
registry = ModelRegistry()

for hosp_name, (X_hosp, y_hosp) in hospitals_data.items():
    client = HospitalClientNode(hosp_name, X_hosp, y_hosp)
    # Train baseline local model (5 epochs)
    client.model = train_model(client.model, X_hosp, y_hosp, epochs=5,
                               classes=np.array([0, 1]))
    clients.append(client)
    registry.register_model(hosp_name, client.model)

server  = FederatedServer()
trainer = FederatedPipelineRunner(clients, server)

# Run 1 initial sync round so we have a valid global model from the start
global_model = trainer.run_training_loop(rounds=1, local_epochs=3)
federated_round = 1

# Track accuracy per round for the chart
accuracy_history: list[float] = []

def _eval_model(model):
    preds = predict(model, X_eval)
    return calculate_metrics(y_eval, preds)

# Initial global accuracy
init_metrics = _eval_model(global_model)
accuracy_history.append(round(init_metrics["Accuracy"] * 100, 2))

print(f"✅ [BOOT] Pipeline ready. Initial global accuracy: {accuracy_history[-1]}%\n")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Apollo HQ Triage Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


# ── Route 1: Network Status ────────────────────────────────────────────────────
@app.get("/api/network-status")
def network_status():
    """Returns each hospital's demographic profile and isolated model accuracy."""
    silos = []
    for client in clients:
        preds = predict(client.model, X_eval)
        m = calculate_metrics(y_eval, preds)
        meta = HOSPITAL_META[client.hospital_id]
        silos.append({
            "id":       client.hospital_id,
            "name":     meta["name"],
            "dominant": meta["dominant"],
            "color":    meta["color"],
            "accuracy": round(m["Accuracy"] * 100, 2),
            "f1":       round(m["F1_Score"], 3),
            "recall":   round(m["Recall"], 3),
        })

    global_m = _eval_model(global_model)
    return {
        "silos":            silos,
        "federated_rounds": federated_round,
        "global_accuracy":  round(global_m["Accuracy"] * 100, 2),
        "global_f1":        round(global_m["F1_Score"], 3),
        "accuracy_history": accuracy_history,
    }


# ── Route 2: Federated Sync ────────────────────────────────────────────────────
@app.post("/api/federated-sync")
def federated_sync():
    """Runs 1 round of FedAvg and returns updated global accuracy."""
    global global_model, federated_round

    global_model  = trainer.run_training_loop(rounds=1, local_epochs=3)
    federated_round += 1

    m = _eval_model(global_model)
    acc = round(m["Accuracy"] * 100, 2)
    accuracy_history.append(acc)

    return {
        "round":    federated_round,
        "accuracy": acc,
        "f1":       round(m["F1_Score"], 3),
        "recall":   round(m["Recall"], 3),
        "precision":round(m["Precision"], 3),
        "accuracy_history": accuracy_history,
    }


# ── Route 3: Predict (Structured Input + XAI) ─────────────────────────────────
class PatientData(BaseModel):
    age: float
    temperature: float
    heart_rate: float
    cough: int           # 0 or 1
    shortness_of_breath: int  # 0 or 1
    travel_history: int  # 0 or 1

@app.post("/api/predict")
def predict_triage(patient: PatientData):
    """
    Takes structured patient vitals, runs prediction through the Federated Global Model,
    and returns urgency + XAI feature importance breakdown.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from phase_1_data.schema import NUMERICAL_FEATURES, FEATURES

    # Build a raw row, then scale numericals to match training distribution
    raw_df = pd.DataFrame([{
        "age": patient.age,
        "temperature": patient.temperature,
        "heart_rate": patient.heart_rate,
        "symptom_cough": patient.cough,
        "symptom_shortness_of_breath": patient.shortness_of_breath,
        "travel_history": patient.travel_history,
    }])

    # Scale using same distribution as the full training data
    scaler = StandardScaler()
    X_train_full, _ = preprocess_data(load_data(n_samples=15000))
    scaler.fit(X_train_full[NUMERICAL_FEATURES])
    raw_df[NUMERICAL_FEATURES] = scaler.transform(raw_df[NUMERICAL_FEATURES])
    X_input = raw_df[FEATURES]

    # Run through global federated model
    urgency      = int(predict(global_model, X_input)[0])
    probabilities= predict_proba(global_model, X_input)[0]

    # ── XAI: Feature Importance via model coefficients ────────────────────────
    coef = global_model.coef_[0]           # shape: (n_features,)
    abs_coef = np.abs(coef)
    total = abs_coef.sum()
    importances = [(FEATURE_NAMES[i], round(float(abs_coef[i] / total * 100), 1))
                   for i in range(len(FEATURE_NAMES))]
    importances.sort(key=lambda x: x[1], reverse=True)

    # Also get individual hospital predictions to show model comparison
    hospital_predictions = []
    for client in clients:
        h_pred   = int(predict(client.model, X_input)[0])
        h_proba  = predict_proba(client.model, X_input)[0]
        hospital_predictions.append({
            "name":       HOSPITAL_META[client.hospital_id]["name"],
            "urgency":    h_pred,
            "confidence": round(float(h_proba[h_pred]) * 100, 1),
        })

    return {
        "urgency":     urgency,
        "urgency_label": "⚠️ HIGH URGENCY" if urgency == 1 else "✅ LOW URGENCY",
        "confidence":  round(float(probabilities[urgency]) * 100, 1),
        "xai_breakdown": importances,
        "hospital_comparison": hospital_predictions,
    }


# ── Route 4: NLP Extraction via Gemini ────────────────────────────────────────
class NLPRequest(BaseModel):
    text: str
    api_key: Optional[str] = None

@app.post("/api/nlp-extract")
def nlp_extract(req: NLPRequest):
    """
    Accepts natural-language patient description.
    Uses Gemini to extract structured triage vitals from free text.
    """
    import json, re

    key = req.api_key or GEMINI_API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API key required.")

    genai.configure(api_key=key)
    model_nlp = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""You are a medical triage data extraction AI.
From the following patient description, extract structured vitals.
Return ONLY a valid JSON object with these exact keys:
- age (number, default 45 if not mentioned)
- temperature (number in Celsius, default 37.0 if not mentioned)
- heart_rate (number in BPM, default 78 if not mentioned)
- cough (0 or 1, 1 if patient has cough)
- shortness_of_breath (0 or 1, 1 if patient has difficulty breathing)
- travel_history (0 or 1, 1 if patient has recent travel history)

Patient description: "{req.text}"

Return ONLY the JSON object, no explanation."""

    response = model_nlp.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Gemini returned invalid JSON: {raw}")

    # Validate and coerce
    extracted = {
        "age":                  float(data.get("age", 45)),
        "temperature":          float(data.get("temperature", 37.0)),
        "heart_rate":           float(data.get("heart_rate", 78)),
        "cough":                int(data.get("cough", 0)),
        "shortness_of_breath":  int(data.get("shortness_of_breath", 0)),
        "travel_history":       int(data.get("travel_history", 0)),
    }
    return {"extracted": extracted, "raw_response": raw}
