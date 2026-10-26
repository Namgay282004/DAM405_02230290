import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import os

# Load test dataset
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("target", axis=1)

# Fetch latest model from MLflow Registry
model_uri = "models:/BestAutoMLModel/latest"
model = mlflow.sklearn.load_model(model_uri)

# Calculate SHAP values
explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)

# Generate and save SHAP Summary Plot
os.makedirs("artifacts", exist_ok=True)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plot_path = "artifacts/shap_summary.png"
plt.savefig(plot_path)
plt.close()

print(f"SHAP summary plot saved successfully to {plot_path}")