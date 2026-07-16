# import joblib
import os
import pandas as pd

import mlflow
import mlflow.pyfunc

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

tracking_uri = os.getenv( "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(tracking_uri)

TEST_DATA_PATH = "data/iris.csv"
TARGET_COLUMN = "species"

# Minimum acceptable performance
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1 = 0.90


def load_model():
    return mlflow.pyfunc.load_model("models:/iris_decision_tree@champion")


def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def test_model_accuracy():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    assert accuracy >= MIN_ACCURACY


def test_model_precision():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"\nPrecision: {precision:.4f}")

    assert precision >= MIN_PRECISION


def test_model_recall():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"\nRecall: {recall:.4f}")

    assert recall >= MIN_RECALL


def test_model_f1():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"\nF1 Score: {f1:.4f}")

    assert f1 >= MIN_F1
