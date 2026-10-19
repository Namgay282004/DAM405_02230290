import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from flaml import AutoML

# Disable FLAML's automatic MLflow hook to prevent skops serialization crashes
os.environ["FLAML_DISABLE_MLFLOW"] = "TRUE"

# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data.values
y = iris.target.values
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training feature baseline means for drift detection in serve.py
np.save('train_feature_means.npy', X_train.mean(axis=0))

# Set MLflow Experiment
mlflow.set_experiment("iris_automl_experiment")

# ---------------------------------------------------------
# 1. Baseline Model Execution & MLflow Tracking
# ---------------------------------------------------------
with mlflow.start_run(run_name="Baseline_RandomForest"):
    baseline_model = RandomForestClassifier(n_estimators=10, random_state=42)
    baseline_model.fit(X_train, y_train)
    preds = baseline_model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    mlflow.log_param("model_type", "Baseline_RandomForest")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(
        baseline_model, 
        "baseline_model", 
        serialization_format="cloudpickle"
    )
    print(f"\n[SUCCESS] Baseline RandomForest - Accuracy: {acc:.4f}, F1: {f1:.4f}")

# ---------------------------------------------------------
# 2. FLAML AutoML Execution & MLflow Tracking
# ---------------------------------------------------------
with mlflow.start_run(run_name="FLAML_AutoML"):
    automl = AutoML()
    automl_settings = {
        "time_budget": 10,  # seconds
        "metric": 'accuracy',
        "task": 'classification',
        "seed": 42,
        "mlflow_logging": False  # Crucial fix: prevents FLAML auto-logging skops error
    }
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    best_model = automl.model.estimator
    automl_preds = automl.predict(X_test)
    
    automl_acc = accuracy_score(y_test, automl_preds)
    automl_f1 = f1_score(y_test, automl_preds, average='weighted')
    
    mlflow.log_param("best_estimator", automl.best_estimator)
    mlflow.log_param("best_config", str(automl.best_config))
    mlflow.log_metric("accuracy", automl_acc)
    mlflow.log_metric("f1_score", automl_f1)
    
    # Save best model artifact locally for serve.py
    joblib.dump(best_model, 'model.joblib')
    
    # Register best model in MLflow Model Registry
    mlflow.sklearn.log_model(
        best_model, 
        "best_model", 
        registered_model_name="IrisAutoMLModel",
        serialization_format="cloudpickle"
    )
    print(f"[SUCCESS] FLAML AutoML - Best Estimator: {automl.best_estimator}, Accuracy: {automl_acc:.4f}")

# ---------------------------------------------------------
# 3. Model Explainability with SHAP
# ---------------------------------------------------------
print("\nGenerating SHAP summary plot...")
try:
    # Standard TreeExplainer for XGBoost/RandomForest models
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    print("[SUCCESS] Saved clean shap_summary.png successfully!\n")
except Exception as e:
    print(f"SHAP note: {e}")