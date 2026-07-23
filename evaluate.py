import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Load data
df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["species"])
y = df["species"]


# Load model from DVC tracked model
model = joblib.load("models/model.joblib")


# Predict
y_pred = model.predict(X)


# Metrics
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