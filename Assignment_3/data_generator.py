import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

os.makedirs("data", exist_ok=True)

# Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=1200,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    random_state=42
)

feature_names = ["age", "income", "credit_score", "loan_amount", "tenure", "debt_ratio"]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Split into train and test
train_df = df.iloc[:1000]
test_df = df.iloc[1000:]

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("Dataset created successfully in ./data/")