# MLOps Weekly Assignment - Week 4

## Overview

This repository contains the implementation of the Week 4 MLOps assignment using GitHub Actions, DVC, Google Cloud Storage (GCS), Continuous Machine Learning (CML), and scikit-learn. The pipeline automatically retrieves versioned datasets and model artifacts from DVC, executes data validation and model evaluation tests, and reports the results on every push and pull request.

## Files

### train.py

* Trains the IRIS classification model.
* Saves the trained model as a Joblib artifact.
* Used for generating the versioned model tracked by DVC.

### tests/test_data_validation.py

* Validates the IRIS dataset before model evaluation.
* Checks dataset schema.
* Verifies the absence of missing values.
* Ensures feature data types are correct.
* Confirms feature values fall within reasonable ranges.
* Validates target class labels.

### tests/test_model_evaluation.py

* Loads the trained model.
* Performs inference on the evaluation dataset.
* Computes evaluation metrics.
* Verifies that model performance satisfies predefined thresholds for:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

### .github/workflows/ci.yml

* Configures the GitHub Actions Continuous Integration pipeline.
* Checks out the repository.
* Installs project dependencies.
* Authenticates with Google Cloud.
* Retrieves versioned datasets and models using DVC.
* Executes the complete pytest test suite.
* Generates a CML report and publishes it as a Pull Request comment.

### data/

* Contains the IRIS dataset tracked using DVC.

### models/

* Contains the trained model tracked using DVC.

### README.md

Provides an overview of the repository and the purpose of each included file.

---

## Technologies Used

* Python
* scikit-learn
* pandas
* pytest
* GitHub Actions
* DVC
* Google Cloud Storage (GCS)
* Continuous Machine Learning (CML)
* Vertex AI Workbench
* joblib

---

## Notes

* Versioned datasets and trained model artifacts are managed using DVC.
* Google Cloud Storage is used as the DVC remote storage.
* GitHub Actions automatically executes the CI pipeline on every push and pull request.
* CML publishes automated test reports as comments on Pull Requests.
* Large datasets and model artifacts are excluded from Git and restored using `dvc pull`.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── iris.csv.dvc
├── models/
│   ├── model.joblib.dvc
├── tests/
│   ├── test_data_validation.py
│   └── test_model_evaluation.py
├── train.py
├── requirements.txt
└── README.md
```

---

## CI Pipeline

```text
Developer Push / Pull Request
            │
            ▼
      GitHub Actions
            │
            ▼
     Authenticate to GCP
            │
            ▼
         DVC Pull
            │
            ▼
      Data Validation Tests
            │
            ▼
    Model Evaluation Tests
            │
            ▼
      Generate CML Report
            │
            ▼
   Comment Results on Pull Request
```

---

## Sample Output

```text
============================= test session starts =============================

tests/test_data_validation.py ........
tests/test_model_evaluation.py ........

======================== 10 passed in 1.76s ========================

Model Evaluation Metrics

Accuracy  : 0.9733
Precision : 0.9732
Recall    : 0.9733
F1 Score  : 0.9732

GitHub Actions Status: PASSED
CML Report: Published successfully on Pull Request
```
