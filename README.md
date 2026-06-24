# MLOps Weekly Assignment - Week 1

## Overview

This repository contains the implementation of the Week 1 MLOps assignment using Google Cloud Platform (GCP), Vertex AI Workbench, Google Cloud Storage (GCS), and scikit-learn.

## Files

### week1_assignment.ipynb
- Uploads IRIS datasets to GCS
- Splits data into training and evaluation sets
- Trains a Decision Tree classifier
- Stores model artifacts and metrics in GCS
- Runs inference on the evaluation set
- Executes the pipeline for multiple runs and multiple dataset versions

### README.md
Provides an overview of the repository and the purpose of each included file.

## Technologies Used

- Python
- scikit-learn
- pandas
- Google Cloud Storage
- Vertex AI Workbench
- joblib

## Notes

- Model artifacts are stored in Google Cloud Storage.
- Binary model files and standard dataset splits and Model artifacts are intentionally excluded from this repository as per the assignment instructions.

## GCS Structure
```text
week-1/
├── data/
└── artifacts/
```

## Sample Output

Training completed successfully.

V1 Accuracy: 0.9667
V2 Accuracy: 0.9500

Artifacts uploaded to:
gs://<bucket-name>/week-1/artifacts/v1/<timestamp>/
gs://<bucket-name>/week-1/artifacts/v2/<timestamp>/

