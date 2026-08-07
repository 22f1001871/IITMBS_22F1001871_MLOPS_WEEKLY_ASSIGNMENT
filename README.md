# MLOps Weekly Assignment - Week 8
## Machine Learning Security: Data Poisoning Attack and Mitigation

## Overview

This project demonstrates the impact of **data poisoning attacks** on a machine learning model using the IRIS dataset. Three poisoned datasets were created by randomly replacing feature values and labels for different percentages of the training data. The performance of a Decision Tree classifier was evaluated on each dataset, and all experiments were tracked using MLflow.

---

## Objectives

- Explain common Machine Learning security threat vectors.
- Simulate data poisoning attacks on the IRIS dataset.
- Measure the impact of poisoned data on model performance.
- Track experiments using MLflow.
- Discuss mitigation strategies for defending production ML pipelines.

---

## Repository Structure

```
.
├── data/
│   ├── iris.csv
│   ├── iris_clean.csv
│   ├── iris_poisoned_5.csv
│   ├── iris_poisoned_10.csv
│   └── iris_poisoned_50.csv
│
├── train.py
├── poison_dataset.py
├── requirements.txt
└── README.md
```

---

## Data Poisoning

The original IRIS dataset is used as the baseline.

Three poisoned datasets were generated:

| Dataset | Corruption |
|----------|------------|
| iris_clean.csv | 0% |
| iris_poisoned_5.csv | 5% |
| iris_poisoned_10.csv | 10% |
| iris_poisoned_50.csv | 50% |

### Poisoning Method

For each selected sample:

- Replace all four features with randomly generated values.
- Assign a random class label.
- Save the modified dataset.

This simulates an attacker injecting noisy samples into the training data.

---

## Model

Algorithm:

- Decision Tree Classifier

Hyperparameters:

- max_depth = 3
- min_samples_split = 2

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

## MLflow Experiment Tracking

Experiment Name:

```
IRIS POISON TESTING
```

Each experiment logs:

- Dataset used
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1-score
- Registered model

The best-performing model is assigned the **champion** alias in the MLflow Model Registry.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Poisoned Datasets

```bash
python poison_dataset.py
```

### Train and Track Experiments

```bash
python train.py
```

---

## Results

The expectation is that increasing the amount of poisoned data reduces model performance.

Typical trend:

| Dataset | Expected Performance |
|----------|----------------------|
| Clean | Highest |
| 5% Poisoned | Slight decrease |
| 10% Poisoned | Moderate decrease |
| 50% Poisoned | Significant decrease |

---

## Security Threats Discussed

- Data Poisoning
- Adversarial Examples
- Model Extraction
- Prompt Injection

---

## Mitigation Strategies

The following defenses are discussed:

- Statistical validation
- Anomaly detection
- Data provenance tracking
- Schema enforcement
- Dataset quality monitoring

---

## Data Quality vs Data Quantity

Key observations:

- High-quality data is more valuable than simply having more data.
- Collecting additional poisoned data does not improve model performance.
- Increasing the proportion of clean data improves model reliability.
- As the clean data ratio decreases, a larger amount of clean data is required to achieve reliable training.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow

---

## Author


- Roll Number: 22F1001871
- Program: BS in Data Science and Applications, IIT Madras

MLOps Weekly Assignment - Week 8
