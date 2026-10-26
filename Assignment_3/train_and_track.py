import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from flaml import AutoML
import lightgbm as lgb
import os

# Set MLflow Experiment
mlflow.set_experiment("DAM405_Assignment_3_AutoML")

# 1. Load Data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
X_test, y_test = test_df.drop("target", axis=1), test_df["target"]

def evaluate_model(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred))
    }

# 2. Baseline Model
with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
    baseline_model = LogisticRegression(random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    
    base_metrics = evaluate_model(y_test, y_pred_base)
    mlflow.log_params({"model_type": "LogisticRegression", "baseline": True})
    mlflow.log_metrics(base_metrics)
    
    # Save baseline model safely using cloudpickle
    mlflow.sklearn.log_model(
        sk_model=baseline_model, 
        artifact_path="baseline_model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    print(f"\n[SUCCESS] Baseline Metrics: {base_metrics}\n")

# 3. FLAML AutoML Model
with mlflow.start_run(run_name="FLAML_AutoML_Run") as run:
    automl = AutoML()
    settings = {
        "time_budget": 30,
        "metric": "f1",
        "task": "classification",
        "seed": 42,
        "log_file_name": "automl.log",
        "mlflow_logging": False  # Disable FLAML's broken internal autologger
    }
    
    # Run AutoML Search
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    
    y_pred_automl = automl.predict(X_test)
    automl_metrics = evaluate_model(y_test, y_pred_automl)
    
    # Convert best config params to standard Python types to avoid log serialization issues
    clean_config = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in automl.best_config.items()}
    
    # Log hyperparameters and metrics to MLflow
    mlflow.log_params({
        "best_estimator": str(automl.best_estimator),
        "time_budget": 30
    })
    mlflow.log_metrics(automl_metrics)
    
    # Extract best estimator underlying model directly
    best_estimator_obj = automl.model.estimator
    
    # Log best estimator explicitly with cloudpickle
    model_info = mlflow.sklearn.log_model(
        sk_model=best_estimator_obj, 
        artifact_path="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    
    # Register Model in MLflow Registry
    mlflow.register_model(model_info.model_uri, "BestAutoMLModel")
    
    print(f"\n[SUCCESS] AutoML Best Estimator: {automl.best_estimator}")
    print(f"[SUCCESS] AutoML Metrics: {automl_metrics}\n")