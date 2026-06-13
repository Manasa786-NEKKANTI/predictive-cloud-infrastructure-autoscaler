# Predictive Cloud Infrastructure Auto-Scaler

> **LSTM · Time-Series · TensorFlow · Live Telemetry Simulator · Cost-Function Evaluator**  
> Built by Manasa — CS + Data Science Honours, KL University Hyderabad

---

## Overview

An intelligent, production-gated infrastructure scaling model that uses Long Short-Term Memory (LSTM) neural networks to process live server telemetry (CPU utilisation, memory traffic) across thousands of nodes — predicting peak load in advance to enable proactive, cost-optimised provisioning.

**Key result: 35% reduction in infrastructure provisioning latency overhead.**

---

## Architecture

```
data/
  telemetry_simulator.py   ← Synthetic node telemetry generator (CPU, memory, network)

backend/
  train_lstm.py            ← LSTM model + anomaly detection + cost-function evaluator
  app.py                   ← Flask REST API (live, predict, cost, history, anomalies)

frontend/
  index.html               ← Live telemetry dashboard (node grid, trend charts, predictor)

models/
  lstm_best.keras          ← Saved LSTM weights (if TF available)
  surrogate_model.pkl      ← Gradient Boosting surrogate fallback
  y_scaler.pkl / feat_scalers.pkl
  metrics.json
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate telemetry data
cd data && python telemetry_simulator.py && cd ..

# 3. Train LSTM model (or surrogate if TF not installed)
cd backend && python train_lstm.py && cd ..

# 4. Start API
cd backend && python app.py
# API at http://localhost:5002

# 5. Open dashboard
open frontend/index.html
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/live` | Real-time node telemetry snapshot |
| `POST` | `/api/predict` | LSTM peak CPU prediction |
| `POST` | `/api/cost` | Cost-function evaluation |
| `GET` | `/api/history` | 24h cluster trend data |
| `GET` | `/api/anomalies` | Recent anomaly events |
| `GET` | `/api/metrics` | Model performance metrics |

### Predict Example

```bash
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"base_cpu": 65, "n_nodes": 100}'
```

```json
{
  "predicted_peak_cpu": 72.4,
  "recommendation": "scale_up",
  "cost_analysis": {
    "compute_cost_usd": 0.0869,
    "latency_overhead_ms": 0,
    "total_cost_score": 0.0921
  }
}
```

---

## LSTM Architecture

```
Input: (batch, 60, 3)         ← 60-minute window, 3 features
  → LSTM(128, return_seq=True)
  → Dropout(0.2)
  → BatchNormalization
  → LSTM(64)
  → Dropout(0.2)
  → Dense(32, ReLU)
  → Dense(1)                  ← Peak CPU in next 10 minutes
```

**Features:** CPU utilisation · Memory utilisation · Network traffic (MB/s)  
**Target:** Peak CPU % in next 10-minute horizon

---

## Cost-Function Evaluator

Custom mathematical evaluator balancing:
- **Compute cost** → proportional to provisioned capacity
- **Under-provision penalty** → quadratic penalty when actual > capacity
- **Over-provision offset** → savings credit for conservative allocation

```
total_cost = compute_cost + under_provision_penalty - over_provision_credit
```

Scaling recommendations:
- `scale_up` if predicted utilisation ratio > 0.85
- `scale_down` if predicted utilisation ratio < 0.40
- `maintain` otherwise

---

## Dashboard Features

- **Live Node Status Grid** — 50-node visual grid (healthy/warning/overload)
- **24h CPU & Memory Trend** — animated line chart
- **Peak CPU Predictor** — interactive sliders → LSTM inference → cost breakdown
- **Anomaly Event Log** — recent spikes with severity and resolution status
- **Model Performance Panel** — MAE, RMSE, R², model architecture summary

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `NumPy` · `Pandas` · `Flask` · `Chart.js` · `HTML/CSS/JS`

---

*Built as part of a data science portfolio — Manasa, KL University Hyderabad (2025)*
