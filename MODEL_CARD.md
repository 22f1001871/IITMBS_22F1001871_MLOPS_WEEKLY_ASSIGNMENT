# Model Card — IRIS Classifier

## 1. Model Overview

This model is a Decision Tree classifier trained on the IRIS dataset.

The model predicts one of three Iris flower species:

- Setosa
- Versicolor
- Virginica

The model uses four input features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The `location` attribute was introduced for fairness analysis but was
not used as a model input.

---

## 2. Intended Use

The model is intended for educational purposes and demonstrates a
machine learning pipeline incorporating fairness assessment,
explainability, and data-drift monitoring.

It should not be considered a production-ready model for real-world
biological or scientific decision-making.

---

## 3. Training Data

The model was trained using the IRIS dataset.

The dataset contains 150 samples belonging to three classes:

- Setosa
- Versicolor
- Virginica

Each sample contains four numerical features:

- Sepal length
- Sepal width
- Petal length
- Petal width

A `location` column containing randomly assigned values of 0 and 1
was added for fairness analysis. This sensitive attribute was excluded
from model training.

---

## 4. Model

Model type: Decision Tree Classifier

Input features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

Sensitive attribute:

- Location (0 or 1)

The location attribute was not used as a predictor.

---

## 5. Overall Performance

The model was evaluated using classification metrics including:

- Accuracy
- Precision
- Recall

The exact evaluation results are reported in the project notebook.

---

## 6. Fairness Analysis

Fairness was evaluated using Fairlearn's `MetricFrame`.

Accuracy, precision, and recall were calculated separately for:

- Location = 0
- Location = 1

Because location was randomly assigned, similar performance across
the two groups was expected.

The results showed the performance of the model across the two location
groups and were used to identify any performance gap.

---

## 7. Explainability

SHAP was used to explain the model predictions.

A full-dataset SHAP explainer was used to generate summary plots for
all three Iris classes.

For the Virginica class, petal length and petal width were the most
influential features.

High values of petal length and petal width generally pushed the model
toward predicting Virginica, while low values generally pushed the
prediction away from Virginica.

Sepal length and sepal width had relatively small SHAP contributions
for Virginica predictions.

---

## 8. Data Drift

Data drift was simulated by shifting the `petal_length` feature in a
simulated production dataset.

The Kolmogorov-Smirnov (KS) test was used to compare the training and
production feature distributions.

The test detected statistically significant drift in `petal_length`,
while no statistically significant drift was detected in the other
three Iris features.

A persistent change in production feature distributions could reduce
model reliability because the model was trained on a different data
distribution.

---

## 9. Limitations

- The dataset is small and contains only 150 samples.
- The model is intended primarily for educational purposes.
- The randomly assigned location attribute does not represent a real
  demographic or protected attribute.
- Fairness results from this artificial location attribute should not
  be interpreted as evidence of fairness in a real-world population.
- The simulated data drift does not represent every type of production
  drift.
- Concept drift was not directly evaluated because the experiment
  focused on changes in feature distributions.
- Model performance may change when the model encounters data outside
  the training distribution.

---

## 10. Fairness Considerations

The `location` attribute was used only for auditing fairness and was
excluded from model training.

Fairness metrics were evaluated separately for the two location
groups. Since location was randomly assigned, substantial systematic
differences between the groups were not expected.

In a real-world application, sensitive attributes should be carefully
defined and monitored, and fairness analysis should consider the
appropriate protected groups and fairness criteria for the application.

---

## 11. Monitoring and Governance

The model should be monitored after deployment for:

- Changes in input feature distributions
- Changes in model performance
- Performance differences across relevant groups
- Changes in the relationship between inputs and predictions

A model card provides documentation of the model's intended use,
training data, performance, fairness considerations, explainability,
and limitations.