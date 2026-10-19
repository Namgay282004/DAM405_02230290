import time
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from flask import Flask, request, jsonify
import numpy as np
import joblib

# ---------------------------------------------------------
# Task 1: Configure Logging
# ---------------------------------------------------------
logger = logging.getLogger('ml-service')
logger.setLevel(logging.INFO)  # Use DEBUG in development

fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Stream Handler (Console output)
console = logging.StreamHandler()
console.setFormatter(fmt)

# Rotating File Handler (Max 1MB, 3 Backups)
fileh = RotatingFileHandler('service.log', maxBytes=1_000_000, backupCount=3)
fileh.setFormatter(fmt)

logger.addHandler(console)
logger.addHandler(fileh)

# ---------------------------------------------------------
# Load Model Artifact & Initialize Flask
# ---------------------------------------------------------
app = Flask(__name__)

try:
    model = joblib.load('model.joblib')
    logger.info('Model successfully loaded from model.joblib')
except Exception as e:
    logger.critical(f'Failed to load model.joblib: {e}')
    raise e

# Baseline training mean for feature 0 (adjust to match your training set baseline)
TRAIN_MEAN_FEATURE_0 = 0.5 

# ---------------------------------------------------------
# Task 4: Record Monitoring Signals
# ---------------------------------------------------------
stats = {'requests': 0, 'errors': 0, 'latencies': deque(maxlen=100)}

def record(latency_ms, ok=True):
    stats['requests'] += 1
    if not ok: 
        stats['errors'] += 1
    stats['latencies'].append(latency_ms)

def summary():
    lat = stats['latencies']
    avg = sum(lat) / len(lat) if lat else 0
    logger.info(f"requests={stats['requests']} errors={stats['errors']} avg_latency_ms={avg:.1f}")

# ---------------------------------------------------------
# Task 5: Basic Drift Check
# ---------------------------------------------------------
def drift_check(train_mean, live_values, tol=0.25):
    live_mean = float(np.mean(live_values))
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)
    if rel > tol:
        logger.warning(f'possible drift: live={live_mean:.3f} train={train_mean:.3f} change={rel:.0%}')
    return rel

# ---------------------------------------------------------
# Task 2: Instrumented Prediction Endpoint
# ---------------------------------------------------------
@app.post('/predict')
def predict():
    start = time.perf_counter()
    try:
        data = request.get_json()
        logger.debug(f'request features: {data}')
        
        # Verify payload structure
        if not data or 'features' not in data:
            raise ValueError("Missing 'features' key in payload")

        features = data['features']
        
        # Drift check on feature 0
        drift_check(TRAIN_MEAN_FEATURE_0, [features[0]])

        # Reshape for model prediction
        x = np.array(features).reshape(1, -1)
        
        # Run inference on loaded joblib model
        raw_pred = model.predict(x)[0]
        pred = int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else float(raw_pred)

        ms = (time.perf_counter() - start) * 1000
        record(ms, ok=True)
        logger.info(f'prediction={pred} latency_ms={ms:.1f}')

        # Log periodic summary every 5 requests
        if stats['requests'] % 5 == 0:
            summary()

        return jsonify({'prediction': pred, 'latency_ms': round(ms, 2)})

    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        record(ms, ok=False)
        logger.error(f'prediction failed: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Task 3 Log Levels demonstration upon startup
    logger.debug('detailed diagnostics for developers')
    logger.info('normal event service handled a request')
    logger.warning('unexpected input but still handled')
    logger.error('a request failed')
    logger.critical('service cannot continue')
    
    app.run(port=5000)