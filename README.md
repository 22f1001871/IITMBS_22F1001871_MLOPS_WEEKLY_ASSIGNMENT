# MLOps Weekly Assignment - Week 5

## Overview

This repository contains the implementation of the Week 5 MLOps assignment using Google Cloud Platform (GCP), MLflow Model Registry, Google Cloud Storage (GCS), GitHub Actions, and Continuous Integration (CI). The pipeline trains multiple Decision Tree models, logs experiments to MLflow, registers the best model, assigns the **champion** alias, and evaluates the registered model during CI.

## Files

### train.py

- Loads the IRIS dataset.
- Splits the data into training and testing sets.
- Trains multiple Decision Tree models with different hyperparameters.
- Logs parameters and evaluation metrics to MLflow.
- Registers each trained model in the MLflow Model Registry.
- Assigns the best-performing model the **champion** alias.

### evaluate.py

- Connects to the remote MLflow Tracking Server.
- Loads the registered model using the **champion** alias.
- Evaluates the model on the IRIS dataset.
- Computes Accuracy, Precision, Recall, and F1 Score.
- Saves the evaluation metrics to `metrics.json`.

### requirements.txt

Contains all Python dependencies required for training, evaluation, MLflow, DVC, and testing.

### data/

Contains the IRIS dataset tracked using DVC.

### tests/

Contains unit tests for:

- Dataset validation
- Model evaluation

### .github/workflows/ci.yml

Implements the GitHub Actions CI pipeline that:

- Sets up Python
- Installs project dependencies
- Authenticates with Google Cloud
- Pulls data from DVC
- Runs automated tests
- Evaluates the registered MLflow model
- Generates a CML report
- Posts the report on Pull Requests

### metrics.json

Stores evaluation metrics generated during model evaluation.

### report.md

Automatically generated CML report containing:

- Evaluation metrics
- Pytest results

## MLflow Features

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Model Registry
- Registered Model Versioning
- Champion Alias
- Remote Artifact Storage in Google Cloud Storage

## Workflow

1. Train multiple Decision Tree models.
2. Log experiments to MLflow.
3. Register models in the MLflow Model Registry.
4. Assign the best model as **champion**.
5. Store model artifacts in Google Cloud Storage.
6. GitHub Actions retrieves the registered model.
7. Evaluate the champion model.
8. Generate and publish the CI report.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- MLflow
- DVC
- Google Cloud Platform (GCP)
- Google Cloud Storage (GCS)
- GitHub Actions
- CML
- Pytest

## Results

The CI pipeline automatically:

- Executes unit tests
- Retrieves the latest champion model from MLflow
- Evaluates the model
- Generates evaluation metrics
- Publishes a markdown report for Pull Requests

