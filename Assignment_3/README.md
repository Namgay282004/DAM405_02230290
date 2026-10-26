# MLOps Pipeline: AutoML, MLflow Tracking, FastAPI Serving & Real-Time Data Drift Monitoring

An end-to-end Machine Learning Operations (MLOps) pipeline built on a synthetic financial risk tabular dataset. The system features automated hyperparameter optimization using FLAML, experiment tracking and model registration via MLflow, global model explainability using SHAP, and real-time model serving via FastAPI with continuous Kolmogorov-Smirnov (KS-test) data drift detection.

---

##  Project Architecture

```text
Data Generation (data_generator.py)
       │
       ▼
Train & Track Pipeline (train_and_track.py)
 ├── Baseline Logistic Regression ──► MLflow Tracking (:5000)
 └── FLAML AutoML Search ───────────► MLflow Model Registry ("BestAutoMLModel" v1)
       │
       ├──► Model Explainability (explain.py) ──► SHAP Summary Plot
       │
       ▼
Production Model Serving (serve_and_monitor.py)
 ├── FastAPI Server (:8080)
 ├── Real-Time KS-Test Data Drift Detection (p < 0.05)
 ├── Audit Logging (logs/app.log)
 └── Telemetry Counter (/metrics)
```

---

##  Prerequisites & Installation

Ensure you have Python 3.9+ installed on your system.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Namgay282004/DAM405_02230290.git
   cd DAM405_02230290/Assignment_3
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Execution Steps

Follow these steps in order to execute the end-to-end pipeline:

### 1. Generate Synthetic Dataset
Generate the continuous tabular dataset split into 1,000 training and 200 test samples:
```bash
python3 data_generator.py
```
*Output files:* `data/train.csv`, `data/test.csv`

---

### 2. Train Models & Track Experiments
Evaluate the Baseline Logistic Regression model, execute FLAML AutoML search, and log metrics/artifacts to MLflow:
```bash
python3 train_and_track.py
```
*Outputs:* Registered model `BestAutoMLModel` (Version 1) in MLflow SQLite store.

To view the MLflow UI dashboard:
```bash
mlflow ui --port 5000
```
Access at: `http://127.0.0.1:5000`

---

### 3. Generate Model Explainability (SHAP)
Compute global feature attributions and generate the SHAP summary plot:
```bash
python3 explain.py
```
*Output artifact:* `artifacts/shap_summary.png`

---

### 4. Serve Model & Monitor Data Drift
Start the FastAPI inference service pulling directly from the MLflow Model Registry:
```bash
python3 serve_and_monitor.py
```
Service endpoints active on `http://0.0.0.0:8080`.

---

##  Testing API Endpoints

Open a new terminal window to test the serving infrastructure.

### Health Check Endpoint
```bash
curl http://localhost:8080/
```
**Response:**
```json
{"status": "healthy"}
```

---

### Prediction Endpoint (Normal Payload)
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 0.5, "income": -0.2, "credit_score": 1.1, "loan_amount": 0.1, "tenure": -0.5, "debt_ratio": 0.3}'
```
**Response:**
```json
{"prediction": 0, "latency_ms": 5.94, "drift_detected": false}
```

---

### Prediction Endpoint (Outlier Payload - Triggers Drift)
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 99.0, "income": 500.0, "credit_score": -100.0, "loan_amount": 99.0, "tenure": 50.0, "debt_ratio": 80.0}'
```
**Response:**
```json
{"prediction": 1, "latency_ms": 7.14, "drift_detected": true}
```

---

### Metrics Endpoint
```bash
curl http://localhost:8080/metrics
```
**Response:**
```json
{"total_requests": 2, "total_errors": 0, "error_rate": 0.0, "average_latency_ms": 6.54}
```

---

##  Repository Structure

```text
Assignment_3/
├── data/
│   ├── train.csv              # Synthetic training dataset
│   └── test.csv               # Synthetic evaluation dataset
├── artifacts/
│   └── shap_summary.png       # SHAP global feature importance plot
├── logs/
│   └── app.log                # Service audit logs with drift alerts
├── data_generator.py          # Synthetic data generation script
├── train_and_track.py         # Baseline, AutoML & MLflow tracking pipeline
├── explain.py                 # SHAP explainability analysis script
├── serve_and_monitor.py       # FastAPI serving & real-time KS drift detector
├── mlflow.db                  # Local SQLite backend store for MLflow
├── requirements.txt           # Project Python dependencies
└── README.md                  # Project documentation
```