import time
import logging
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import ks_2samp
import os

# 1. Setup Structured Python Logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("MLOps_Service")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')

# File Handler
file_handler = logging.FileHandler("logs/app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 2. Load Model & Training Data Baseline
model_uri = "models:/BestAutoMLModel/latest"
try:
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Successfully loaded production model from MLflow Registry.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")

reference_df = pd.read_csv("data/train.csv").drop("target", axis=1)

# Request Counter and Metrics
metrics_store = {
    "total_requests": 0,
    "total_errors": 0,
    "latencies": []
}

app = FastAPI(title="MLOps Monitored Model Service")

class PredictionInput(BaseModel):
    age: float
    income: float
    credit_score: float
    loan_amount: float
    tenure: float
    debt_ratio: float

@app.get("/")
def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    start_time = time.time()
    metrics_store["total_requests"] += 1
    
    try:
        input_data = pd.DataFrame([data.dict()])
        
        # Data Drift Detection (KS-Test)
        drift_detected = False
        for col in input_data.columns:
            stat, p_value = ks_2samp(reference_df[col], input_data[col])
            if p_value < 0.05:  # Drift threshold
                logger.warning(f"DRIFT DETECTED in feature '{col}'! p-value: {p_value:.4f}")
                drift_detected = True
        
        # Prediction
        prediction = int(model.predict(input_data)[0])
        latency = round((time.time() - start_time) * 1000, 2)
        metrics_store["latencies"].append(latency)
        
        logger.info(f"Prediction: {prediction} | Latency: {latency}ms | Drift Flag: {drift_detected}")
        
        return {
            "prediction": prediction,
            "latency_ms": latency,
            "drift_detected": drift_detected
        }
    
    except Exception as e:
        metrics_store["total_errors"] += 1
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Model Error")

@app.get("/metrics")
def get_metrics():
    avg_latency = np.mean(metrics_store["latencies"]) if metrics_store["latencies"] else 0.0
    error_rate = (metrics_store["total_errors"] / metrics_store["total_requests"]) if metrics_store["total_requests"] > 0 else 0.0
    
    metrics_summary = {
        "total_requests": metrics_store["total_requests"],
        "total_errors": metrics_store["total_errors"],
        "error_rate": error_rate,
        "average_latency_ms": round(avg_latency, 2)
    }
    logger.info(f"Metrics queried: {metrics_summary}")
    return metrics_summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)