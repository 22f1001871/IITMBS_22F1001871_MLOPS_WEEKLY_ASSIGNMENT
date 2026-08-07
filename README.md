# MLOps Weekly Assignment - Week 6
## Dockerization and Kubernetes Deployment of the IRIS Classification API

## Overview

This repository contains the implementation of the Week 6 MLOps assignment. The objective of this assignment is to containerize the IRIS prediction API using Docker, deploy it to a Kubernetes cluster on Google Kubernetes Engine (GKE), and expose the service for online inference.

The application is built using FastAPI and serves predictions from a trained Decision Tree model.

---

## Project Structure

```
.
├── app.py                  # FastAPI application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── deployment.yaml         # Kubernetes Deployment
├── service.yaml            # Kubernetes Service
├── models/
│   └── model.joblib        # Trained model
├── data/
│   └── iris.csv            # Dataset
└── README.md
```

---

## Features

- FastAPI REST API for IRIS flower classification
- Dockerized application
- Kubernetes deployment on Google Kubernetes Engine (GKE)
- External LoadBalancer service
- Scalable deployment architecture
- Ready for Continuous Deployment workflows

---

## Technologies Used

- Python 3.12
- FastAPI
- Scikit-learn
- Joblib
- Docker
- Kubernetes
- Google Kubernetes Engine (GKE)
- Google Artifact Registry

---

## Dockerization

The application is containerized using Docker.

### Build Docker Image

```bash
docker build -t iris-api:v1 .
```

### Run Locally

```bash
docker run -p 5000:5000 iris-api:v1
```

The API will be available at

```
http://localhost:5000
*will vary based on the IP
```

Swagger documentation:

```
http://localhost:5000/docs
```

---

## Push Image to Artifact Registry

Tag the image

```bash
docker tag iris-api:v1 REGION-docker.pkg.dev/PROJECT_ID/iris-repo/iris-api:v1
```

Push image

```bash
docker push REGION-docker.pkg.dev/PROJECT_ID/iris-repo/iris-api:v1
```

---

## Kubernetes Deployment

Create the deployment

```bash
kubectl apply -f deployment.yaml
```

Create the service

```bash
kubectl apply -f service.yaml
```

Verify deployment

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

---

## API Endpoints

### Home

```
GET /
```

Returns a welcome message.

---

### Predict

```
POST /predict
```

Example request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response

```json
{
  "prediction": "setosa"
}
```

---

## Deployment Verification

Once the service is deployed, obtain the external IP:

```bash
kubectl get services
```

Example request

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
-H "Content-Type: application/json" \
-d '{
  "sepal_length":5.1,
  "sepal_width":3.5,
  "petal_length":1.4,
  "petal_width":0.2
}'
```

---

## Model

The deployed model is a Decision Tree Classifier trained on the IRIS dataset.

Target classes:

- Setosa
- Versicolor
- Virginica

---

## Learning Outcomes

This assignment demonstrates:

- Building Docker images for machine learning applications
- Containerizing FastAPI services
- Deploying containers to Kubernetes
- Exposing applications using LoadBalancer services
- Managing containerized ML inference workloads
- Preparing ML applications for production deployment

---

## Future Improvements

- Horizontal Pod Autoscaling (HPA)
- CI/CD using GitHub Actions
- MLflow Model Registry integration
- Monitoring with Prometheus and Grafana
- Canary deployments
- Rolling updates

---

## Author

- Roll Number: 22F1001871
- Program: BS in Data Science and Applications, IIT Madras

MLOps Weekly Assignment - Week 6
