# Script to train the model

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Load dataset
data = pd.read_csv("data/iris.csv")

print("Info of the dataset")
print("-------------------")
data.info()
print("-------------------")

# Features and Target
X = data.drop(columns=["species"])
y = data["species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Create models directory
os.makedirs("models", exist_ok=True)


# Train model
model = DecisionTreeClassifier(max_depth=3,min_samples_split=2,random_state=42)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")



# # Save model locally 
joblib.dump(model, "models/model.joblib")
print("Model saved successfully to models/model.joblib")
