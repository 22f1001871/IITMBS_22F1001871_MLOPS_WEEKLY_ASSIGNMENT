# MLOps Weekly Assignment - Week 2

## Overview

This repository contains the implementation of the Week 2 MLOps assignment using Google Cloud Platform (GCP), Clould Shell Editor, Google Cloud Storage (GCS), GitHub amd DVC.

## Files

## Files

### train.py
- Loads the IRIS dataset
- Trains a Random Forest classifier
- Saves the trained model as a Joblib artifact

### iris.csv.dvc
- DVC pointer file for the IRIS dataset
- Tracks dataset versions without storing the actual dataset in Git

### model.joblib.dvc
- DVC pointer file for the trained model artifact
- Enables model versioning through DVC

### requirements.txt
Contains the Python dependencies required to run the project.

### Week-2.pdf
Contains shell commands required to complete the assignment.

### README.md
Provides an overview of the repository and the purpose of each included file.

## Technologies Used

- Python
- scikit-learn
- pandas
- DVC
- Google Cloud Storage
- Git
- joblib

## Notes

- Actual datasets and model artifacts are stored in Google Cloud Storage through DVC.
- Git stores only lightweight `.dvc` pointer files.
- Three dataset/model iterations were created and versioned during the assignment.
- Previous versions can be restored using Git checkout and DVC checkout.

## Repository Structure

```text
.
├── .dvc/
├── .dvcignore
├── .gitignore
├── README.md
├── .Week-2.pdf
├── requirements.txt
├── train.py
├── iris.csv.dvc
└── model.joblib.dvc
```

## DVC Remote Storage

```text
GCS Bucket
└── week-2/
    └── dvc/
```

## Sample Workflow

```bash
python train.py

dvc add iris.csv
dvc add model.joblib

git add .
git commit -m "Iteration 1"

dvc push
git push origin week_2

#command to tag
git tag -a "v1.0" -m "Iteration 1 model"
git tag -a "v2.0" -m "Iteration 2 model"
git tag -a "v3.0" -m "Iteration 3 model"

```

## Learning Outcomes

- Initialized DVC in a Git repository.
- Configured Google Cloud Storage as a DVC remote.
- Versioned datasets and model artifacts across multiple iterations.
- Stored large files outside Git while maintaining version history.
- Restored previous dataset and model versions using Git and DVC.
