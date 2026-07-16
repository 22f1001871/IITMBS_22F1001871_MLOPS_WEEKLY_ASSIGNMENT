import os
import json
# import joblib
import pandas as pd

import mlflow 
import mlflow.pyfunc

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

tracking_uri = os.getenv( "MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(tracking_uri)

# Load data
df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["species"])
y = df["species"]

# Load model
# model = joblib.load("models/model.joblib")
model = mlflow.pyfunc.load_model("models:/iris_decision_tree@champion")

# Predict
y_pred = model.predict(X)

# Compute metrics
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, average="macro"),
    "recall": recall_score(y, y_pred, average="macro"),
    "f1_score": f1_score(y, y_pred, average="macro"),
}

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)