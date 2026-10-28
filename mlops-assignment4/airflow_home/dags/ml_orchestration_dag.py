from datetime import datetime, timedelta
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'start_date': datetime(2026, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Live AWS API Gateway URL deployed from your CloudFormation stack
API_GATEWAY_URL = "https://9drhgjpqhi.execute-api.ap-south-1.amazonaws.com/Prod/predict/"

def extract_data(**kwargs):
    dataset = [
        {"id": 101, "features": [0.82, 0.64, 0.91]},
        {"id": 102, "features": [0.12, 0.25, 0.33]},
        {"id": 103, "features": [0.75, 0.58, 0.69]}
    ]
    print(f"Extracted {len(dataset)} records.")
    return dataset

def validate_data_quality(**kwargs):
    """Data Quality Gate Task: Verifies features exist and are inside [0, 1] range."""
    ti = kwargs['ti']
    extracted = ti.xcom_pull(task_ids='extract_data')
    if not extracted:
        raise ValueError("Quality Gate Failed: No data found.")
        
    for item in extracted:
        feats = item.get("features", [])
        if not feats:
            raise ValueError(f"Quality Gate Failed: Empty features in record {item.get('id')}.")
        if any(f < 0.0 or f > 1.0 for f in feats):
            raise ValueError("Quality Gate Failed: Feature value out of [0, 1] range.")
            
    print("Data-Quality Gate: All checks passed.")
    return True

def transform_data(**kwargs):
    ti = kwargs['ti']
    extracted = ti.xcom_pull(task_ids='extract_data')
    features_list = [record["features"] for record in extracted]
    return features_list

def trigger_lambda_inference(**kwargs):
    ti = kwargs['ti']
    all_features = ti.xcom_pull(task_ids='transform_data')
    payload = {"features": all_features[0]}
    
    print(f"Invoking AWS Lambda endpoint with payload: {payload}")
    response = requests.post(API_GATEWAY_URL, json=payload, timeout=15)
    
    print(f"Status Code: {response.status_code}")
    print(f"AWS Response: {response.text}")
    
    if response.status_code != 200:
        raise Exception(f"Inference failed with status {response.status_code}")

with DAG(
    'ml_orchestration_pipeline',
    default_args=default_args,
    description='MLOps Pipeline with Data Quality Gate & Serverless Inference',
    schedule='@daily',
    catchup=False,
) as dag:

    task_extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )

    task_validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality
    )

    task_transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )

    task_infer = PythonOperator(
        task_id='trigger_lambda_inference',
        python_callable=trigger_lambda_inference
    )

    # Task dependency graph
    task_extract >> task_validate >> task_transform >> task_infer