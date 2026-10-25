# MLOps Programming Assignment 4: Serverless Model Deployment & Pipeline Orchestration on AWS

This repository contains the complete implementation for **DAM405 Machine Learning Operations Programming Assignment 4**. The project demonstrates an end-to-end MLOps workflow combining serverless model deployment via AWS SAM and pipeline orchestration using Apache Airflow with an embedded Data-Quality Gate.

---

## System Architecture & Workflow

1. **Ingestion & Orchestration (Apache Airflow)**:
   - `extract_data`: Pulls batch feature vectors.
   - `validate_data_quality`: Executes data quality guardrails, enforcing numerical range validation ($0.0 \le f \le 1.0$).
   - `transform_data`: Formats validated features into JSON payloads.
   - `trigger_lambda_inference`: Sends an HTTP POST payload to the API Gateway REST endpoint.

2. **Serverless Execution (AWS Infrastructure - `ap-south-1`)**:
   - **Amazon API Gateway**: Exposes public REST API route `/Prod/predict/`.
   - **AWS Lambda**: Executes Python model inference and returns class prediction and confidence scores.
   - **Amazon CloudWatch**: Captures logs, latencies, and execution telemetry.

---

## Repository Structure

```text
mlops-assignment4/
├── dags/
│   └── ml_orchestration_dag.py     # Apache Airflow DAG definition
├── lambda/
│   └── app.py                      # AWS Lambda model inference handler
├── iac/
│   ├── template.yaml               # AWS SAM Infrastructure as Code
│   └── samconfig.toml              # SAM deployment configuration
├── data/                           # Batch data directory
├── tests/                          # Unit and pipeline tests
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Setup & Deployment Instructions

### Prerequisites
- AWS CLI configured with valid credentials (`ap-south-1` region)
- AWS SAM CLI installed
- Python 3.11+
- Apache Airflow 2.x / 3.x installed

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/mlops-assignment4.git
cd mlops-assignment4
```

---

### Step 2: Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Step 3: Deploy Serverless Infrastructure (AWS SAM)
```bash
cd iac/
sam build
sam deploy --guided
```
*Note: Copy the `InferenceApiEndpoint` output URL produced after deployment.*

---

### Step 4: Test the Live Inference Endpoint
```bash
curl -X POST <YOUR_API_GATEWAY_URL> \
  -H "Content-Type: application/json" \
  -d '{"features": [0.8, 0.7, 0.9]}'
```

**Expected Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "confidence_score": 0.8
}
```

---

### Step 5: Run Apache Airflow DAG
1. Copy or link `dags/ml_orchestration_dag.py` into your Airflow DAGs folder.
2. Ensure `API_GATEWAY_URL` in `ml_orchestration_dag.py` matches your deployed endpoint.
3. Trigger and test the pipeline:
```bash
airflow dags test ml_orchestration_pipeline 2026-10-23
```

---

## Data Quality Gate Rules
The pipeline enforces the following checks inside `validate_data_quality`:
1. Payload presence check (data non-empty).
2. Feature key existence check.
3. Feature value boundary check ($0.0 \le f \le 1.0$). If any value falls outside this range, the pipeline aborts and marks the task as failed.