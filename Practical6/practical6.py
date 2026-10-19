import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Wine dataset
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow Experiment
mlflow.set_experiment('wine-classifier')

def run_experiment(n_estimators, max_depth):
    """Tracks training runs with hyperparameter logging."""
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        ).fit(Xtr, ytr)
        
        acc = accuracy_score(yte, clf.predict(Xte))
        
        # Task 2: Log parameters, metrics, and model
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(
            clf, 
            'model', 
            serialization_format="cloudpickle"
        )
        
        print(f'n={n_estimators} depth={max_depth} acc={acc:.4f} run_id={run.info.run_id}')

# Task 3: Run several hyperparameter experiments
print("Starting MLflow hyperparameter search runs...")
for n in (50, 100, 200):
    for d in (3, 5, None):
        run_experiment(n, d)

# Task 5: Find and Register the best model programmatically
experiment = mlflow.get_experiment_by_name('wine-classifier')
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
best_run_id = runs.iloc[0]["run_id"]
best_acc = runs.iloc[0]["metrics.accuracy"]

print(f"\n[BEST RUN] ID: {best_run_id} with Accuracy: {best_acc:.4f}")

# Register best model in MLflow Model Registry
model_uri = f"runs:/{best_run_id}/model"
reg_model = mlflow.register_model(model_uri, "wine-classifier")
print(f"[SUCCESS] Registered Model Version: {reg_model.version}")

# Task 6: Load the registered model and predict
print("\nLoading registered model from registry...")
loaded_model = mlflow.sklearn.load_model('models:/wine-classifier/latest')
predictions = loaded_model.predict(Xte[:5])
print('Loaded model predictions for top 5 samples:', predictions)