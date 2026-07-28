# MLOps Weekly Assignment - Week 7

## Overview

This repository contains the implementation of the Week 7 MLOps assignment, extending the CI/CD pipeline with automated stress testing, Kubernetes Horizontal Pod Autoscaling (HPA), and monitoring using Google Cloud Platform (GCP). The project deploys a Flask-based IRIS classification API on Google Kubernetes Engine (GKE), performs automated deployment through GitHub Actions, executes load testing using `wrk`, and demonstrates autoscaling under varying workloads.

---

## Files

### app.py

* Implements the Flask-based IRIS prediction API.
* Loads the trained Decision Tree model.
* Provides the `/predict` endpoint for inference.

### evaluate.py

* Evaluates the trained model.
* Computes Accuracy, Precision, Recall, and F1 Score.
* Generates `metrics.json` for the CI pipeline.

### tests/

#### tests/test_data_validation.py

* Validates the IRIS dataset.
* Checks for missing values.
* Verifies feature data types.
* Confirms valid target labels.

#### tests/test_model_evaluation.py

* Loads the trained model.
* Performs inference on the evaluation dataset.
* Verifies that evaluation metrics satisfy predefined thresholds.

### Dockerfile

* Builds the Docker image for the Flask inference API.
* Packages the application and required dependencies for deployment.

### k8s/deployment.yaml

* Defines the Kubernetes Deployment for the IRIS API.
* Configures the container image and resource requests/limits.

### k8s/service.yaml

* Exposes the application using a Kubernetes LoadBalancer Service.
* Provides an external endpoint for client requests.

### k8s/hpa.yaml

* Configures the Horizontal Pod Autoscaler (HPA).
* Scales the deployment based on CPU utilization.
* Uses:

  * `minReplicas: 1`
  * `maxReplicas: 3`
  * CPU utilization target of 50%.

### .github/workflows/ci_cd.yml

Implements the complete CI/CD pipeline using GitHub Actions.

Pipeline stages include:

* Repository checkout
* Python environment setup
* Dependency installation
* Google Cloud authentication
* DVC data and model retrieval
* Automated testing with Pytest
* Model evaluation
* CML report generation
* Docker image build
* Push to Google Artifact Registry
* Deployment to Google Kubernetes Engine
* Deployment verification
* Automated stress testing using `wrk`

### data/

* Contains the IRIS dataset tracked using DVC.

### models/

* Contains the trained Decision Tree model tracked using DVC.

### README.md

Provides documentation for the Week 7 MLOps pipeline.

---

## Technologies Used

* Python
* Flask
* scikit-learn
* pandas
* NumPy
* joblib
* pytest
* Docker
* Kubernetes
* Google Kubernetes Engine (GKE)
* Google Artifact Registry
* Google Cloud Storage (GCS)
* GitHub Actions
* DVC
* Continuous Machine Learning (CML)
* `wrk`
* Horizontal Pod Autoscaler (HPA)
* Google Cloud Monitoring
* Google Cloud Logging

---

## Week 7 Features

### Continuous Integration

* Executes automated tests using Pytest.
* Retrieves datasets and model artifacts from DVC.
* Generates model evaluation metrics.
* Publishes a CML report on Pull Requests.

### Continuous Deployment

* Builds a Docker image.
* Pushes the image to Google Artifact Registry.
* Deploys the latest image to Google Kubernetes Engine.
* Verifies successful deployment.

### Stress Testing

* Installs `wrk` during the GitHub Actions workflow.
* Executes a stress test against the deployed API.
* Reports Requests per Second, Latency, and Error Count.

### Horizontal Pod Autoscaling

* Automatically scales pods according to CPU utilization.
* Demonstrates scaling from one pod to three pods during high load.

### Monitoring

* Uses Google Cloud Monitoring to observe:

  * CPU utilization
  * Memory utilization
  * Pod scaling
* Uses Google Cloud Logging to inspect application logs during stress testing.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── data/
│   └── iris.csv.dvc
├── models/
│   └── model.joblib.dvc
├── tests/
│   ├── test_data_validation.py
│   └── test_model_evaluation.py
├── app.py
├── evaluate.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## CI/CD Pipeline

```text
Developer Push
       │
       ▼
GitHub Actions
       │
       ▼
Install Dependencies
       │
       ▼
Authenticate to Google Cloud
       │
       ▼
DVC Pull
       │
       ▼
Run Pytest
       │
       ▼
Evaluate Model
       │
       ▼
Generate CML Report
       │
       ▼
Build Docker Image
       │
       ▼
Push to Artifact Registry
       │
       ▼
Deploy to GKE
       │
       ▼
Verify Deployment
       │
       ▼
Install wrk
       │
       ▼
Stress Test (1000 Connections)
       │
       ▼
Application Available on GKE
```

---

## Horizontal Pod Autoscaling Workflow

```text
wrk Load Test
       │
       ▼
CPU Utilization Increases
       │
       ▼
Horizontal Pod Autoscaler
       │
       ▼
Scale Pods
1  ─────► 3
       │
       ▼
Reduced CPU Utilization
       │
       ▼
Improved Throughput
```

---

## Monitoring

During stress testing, Google Cloud Monitoring and Cloud Logging are used to observe:

* CPU utilization
* Memory utilization
* Active pod replicas
* Application container logs
* Autoscaling behaviour

---

## Notes

* Datasets and trained model artifacts are versioned using DVC.
* Google Cloud Storage is used as the DVC remote.
* Docker images are stored in Google Artifact Registry.
* The application is deployed on Google Kubernetes Engine.
* Horizontal Pod Autoscaler dynamically scales pods based on CPU utilization.
* GitHub Actions performs end-to-end CI/CD and executes automated stress testing after deployment.
