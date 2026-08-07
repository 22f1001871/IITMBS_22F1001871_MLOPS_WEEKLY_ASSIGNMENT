# Script to train the model and log experiments with MLflow

import os
import pandas as pd
#import joblib

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://34.61.98.9:5000/" 
)

mlflow.set_tracking_uri(tracking_uri)

print("Tracking URI:", mlflow.get_tracking_uri())

# Create MLflow experiment
mlflow.set_experiment("IRIS POISON TESTING")


# Hyperparameter combinations
param_grid = {"max_depth": 3, "min_samples_split": 2}

data_grid = ["data/iris_clean.csv", "data/iris_poisoned_5.csv", "data/iris_poisoned_10.csv", "data/iris_poisoned_50.csv" ]

best_model = None
best_accuracy = 0

client = MlflowClient()

for i, data_path in enumerate(data_grid, start=1):

    data = pd.read_csv(data_path)

    
    print("Info of the dataset")
    print("-------------------")
    data.info()
    print("-------------------")
    
    # Features and Target
    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    
    #Create models directory
    os.makedirs("models", exist_ok=True)

    print(f"\n========== Experiment {i} ==========")
    
    with mlflow.start_run(run_name=f"Experiment_{i}_{os.path.basename(data_path).replace('.csv', '')}"):

        mlflow.log_param("dataset", os.path.basename(data_path))

        # Train model
        model = DecisionTreeClassifier(
            max_depth=param_grid["max_depth"],
            min_samples_split=param_grid["min_samples_split"],
            random_state=42,
        )

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

        # Log Parameters
        mlflow.log_param("max_depth", param_grid["max_depth"])
        mlflow.log_param("min_samples_split", param_grid["min_samples_split"])

        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # # Save model locally (Temporary - will remove in Task 4)
        # joblib.dump(model, f"models/model_{i}.joblib")

        # Log model to MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
             registered_model_name="iris_decision_tree",
        )

        # Keep track of best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

            model_version = model_info.registered_model_version
            best_version = model_version

# Save best model locally (Temporary - will remove in Task 4)
# joblib.dump(best_model, "models/model.joblib")

client.set_registered_model_alias(
    name="iris_decision_tree",
    alias="champion",
    version=best_version,
)

print(f"Champion alias assigned to verison {best_version}")

print("\n===================================")
print("Training Complete!")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("===================================")