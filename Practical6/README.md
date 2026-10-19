# Practical 6: Managing the ML Lifecycle with MLflow

This repository contains the solution for **Practical 6: Managing the ML Lifecycle with MLflow** under the course **DAM405 Machine Learning Operations**.

## Overview
The goal of this project is to manage the machine learning lifecycle by tracking experiments, comparing model runs, registering the best model in the MLflow Model Registry, and loading it back for inference on the Wine dataset.

## Features
- **Experiment Tracking:** Trains 9 Random Forest classifiers across combinations of `n_estimators` (50, 100, 200) and `max_depth` (3, 5, None).
- **Metric Logging:** Logs hyperparameters, test accuracy, and model binaries to MLflow.
- **Model Registration:** Programmatically selects the best-performing model (`accuracy = 1.0000`) and registers it in the MLflow Model Registry as `wine-classifier`.
- **Inference Loading:** Loads the latest registered model (`models:/wine-classifier/latest`) and executes predictions.

## Prerequisites
Ensure Python 3.8+ is installed. Install the required dependencies:

```bash
pip install mlflow scikit-learn

```

## How to Run

1. **Run the MLflow Training & Registration Pipeline:**
```bash
python3 practical6.py

```


*This script trains all 9 model configurations, registers the best model version, and outputs inference test predictions.*
2. **Launch the MLflow Tracking UI:**
```bash
mlflow ui --port 5000

```


3. **View the Dashboard:**
Open your browser and navigate to `http://localhost:5000` to view:
* Experiment runs and hyperparameter comparisons under **Model training > Experiments > wine-classifier**.
* Saved model artifacts under **Model training > Models**.
* Registered model versions under **Model registry**.



## Repository Structure

```text
├── practical6.py    # Main script for experiment tracking, registration, and inference
├── mlruns/          # MLflow local tracking store directory
└── README.md        # Project documentation

```