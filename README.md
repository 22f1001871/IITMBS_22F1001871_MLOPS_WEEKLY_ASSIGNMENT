# MLOps Weekly Assignment - Week 9
# Explainability, Fairness, and Drift in the IRIS Pipeline

## Overview

This assignment extends the IRIS machine learning pipeline with
explainability, fairness analysis, data drift detection, and model
governance.

The following concepts were implemented:

- Sensitive attribute introduction
- Fairness evaluation using Fairlearn
- Model explainability using SHAP
- Data drift detection using the Kolmogorov-Smirnov (KS) test
- Model documentation using a Model Card

The experiments were performed using a Jupyter Notebook.

---

## Dataset

The IRIS dataset was used for training and evaluation.

The dataset contains 150 samples belonging to three classes:

- Setosa
- Versicolor
- Virginica

The four original features are:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

A `location` attribute was additionally introduced for fairness
analysis.

---

## Model

A Decision Tree Classifier was used for classification.

The model was trained using only the original four Iris features:

```text
sepal_length
sepal_width
petal_length
petal_width