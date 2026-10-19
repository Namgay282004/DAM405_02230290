import time
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from flask import Flask, request, jsonify
import numpy as np
import joblib

# =========================================================
# Task 1: Configure Structured Dual Logging
# =========================================================
logger = logging.getLogger('ml-service')
logger.setLevel(logging.INFO)  # Set to DEBUG during local dev if needed

fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Console stdout Handler
console = logging.StreamHandler()
console.setFormatter(fmt)

# Persistent Rotating File Handler (Max 1MB, 3 backup files)
fileh = RotatingFileHandler('service.log', maxBytes=1_000_000, backupCount=3)
fileh.setFormatter(fmt)

logger.addHandler(console)
logger.addHandler(fileh)

# =========================================================
# Service Initialization & Artifact Loading
# =========================================================
app = Flask(__name__)

# Load Model Artifact saved by FLAML AutoML in train.py
try:
    model = joblib.load('model.joblib')
    logger.info("Model successfully loaded from model.joblib")
except Exception as e:
    logger.critical(f"Failed to load model.joblib artifact: {e}")
    raise e

# Load Baseline Training Feature Means for Drift Detection
try:
    train_means = np.load('train_feature_means.npy')
    TRAIN_MEAN_FEATURE_0 = float(train_means[0])
    logger.info(f"Loaded training mean for Feature 0 baseline: {TRAIN_MEAN_FEATURE_0:.4f}")
except Exception as e:
    TRAIN_MEAN_FEATURE_0 = 5.84  # Iris Sepal Length training mean fallback
    logger.warning(f"Could not load train_feature_means.npy ({e}), using fallback: {TRAIN_MEAN_FEATURE_0}")

# =========================================================
# Operational Monitoring & Telemetry Counters
# =========================================================
stats = {'requests': 0, 'errors': 0, 'latencies': deque(maxlen=100)}

def record(latency_ms, ok=True):
    """Tracks cumulative request volume, error counts, and rolling latencies."""
    stats['requests'] += 1
    if not ok:
        stats['errors'] += 1
    stats['latencies'].append(latency_ms)

def summary():
    """Publishes a telemetry summary metric line every 5 requests."""
    lat = stats['latencies']
    avg = sum(lat) / len(lat) if lat else 0
    logger.info(f"requests={stats['requests']} errors={stats['errors']} avg_latency_ms={avg:.1f}")

# =========================================================
# Feature Data Drift Detection Check
# =========================================================
def drift_check(train_mean, live_values, tol=0.25):
    """Flags when live feature average shifts beyond tolerance limit (25%)."""
    live_mean = float(np.mean(live_values))
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)
    if rel > tol:
        logger.warning(f"possible drift: live={live_mean:.3f} train={train_mean:.3f} change={rel:.0%}")
    return rel

# =========================================================
# Prediction Serving Endpoint
# =========================================================
@app.post('/predict')
def predict():
    start = time.perf_counter()
    try:
        data = request.get_json()
        logger.debug(f"request features: {data}")

        if not data or 'features' not in data:
            raise ValueError("Missing 'features' key in request payload")

        features = data['features']

        # Perform data-drift evaluation on Primary Feature 0
        drift_check(TRAIN_MEAN_FEATURE_0, [features[0]])

        # Reshape input array for inference
        x = np.array(features).reshape(1, -1)
        raw_pred = model.predict(x)[0]
        pred = int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else float(raw_pred)

        ms = (time.perf_counter() - start) * 1000
        record(ms, ok=True)
        logger.info(f"prediction={pred} latency_ms={ms:.1f}")

        # Trigger telemetry metric summary line every 5 requests
        if stats['requests'] % 5 == 0:
            summary()

        return jsonify({'prediction': pred, 'latency_ms': round(ms, 2)})

    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        record(ms, ok=False)
        logger.error(f"prediction failed: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Demonstrate log levels upon startup
    logger.debug('detailed diagnostics for developers')
    logger.info('normal event service handled a request')
    logger.warning('unexpected input but still handled')
    logger.error('a request failed')
    logger.critical('service cannot continue')

    # Running on port 5001 to avoid conflict with MLflow UI on 5000
    app.run(port=5001)