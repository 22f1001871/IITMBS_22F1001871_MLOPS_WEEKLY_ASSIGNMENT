import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import vertexai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vertexai.generative_models import GenerativeModel


VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica",
}


def build_v1_prompt(row):
    return (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )


def build_v2_prompt(row):
    return (
        f"A flower specimen has a sepal length of "
        f"{row['sepal_length']} cm, "
        f"sepal width of {row['sepal_width']} cm, "
        f"petal length of {row['petal_length']} cm, "
        f"and petal width of {row['petal_width']} cm. "
        f"Identify the iris species."
    )


def extract_species(response):
    """
    Extract a species name from the model response.

    This is used for classification accuracy.

    Format compliance is calculated separately and requires
    an exact valid species name.
    """

    if response is None:
        return None

    text = response.strip().lower()

    for species in VALID_SPECIES:
        if re.search(rf"\b{species}\b", text):
            return species

    return None


def is_format_compliant(response):
    """
    Assignment definition:

    A response is compliant only when it is exactly:

        setosa
        versicolor
        virginica

    No extra text, JSON, punctuation, or wrong casing.
    """

    if response is None:
        return False

    return response.strip() in VALID_SPECIES


def predict(model, prompt):
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 500,
        },
    )

    if response.text is None:
        return ""

    return response.text.strip()


def evaluate_model(
    model,
    df,
    version,
):
    records = []

    y_true = []
    y_pred = []

    compliant_count = 0

    for index, row in df.iterrows():

        expected = str(row["species"]).strip().lower()

        if version == "v1":
            prompt = build_v1_prompt(row)
        else:
            prompt = build_v2_prompt(row)

        try:
            response = predict(model, prompt)

        except Exception as exc:
            response = ""
            print(
                f"ERROR on sample {index}: {exc}"
            )

        predicted_species = extract_species(response)

        compliant = is_format_compliant(response)

        if compliant:
            compliant_count += 1

        y_true.append(expected)

        # Unknown/malformed responses are treated
        # as an incorrect classification.
        if predicted_species in VALID_SPECIES:
            y_pred.append(predicted_species)
        else:
            y_pred.append("__invalid__")

        records.append(
            {
                "index": int(index),
                "prompt": prompt,
                "expected_species": expected,
                "raw_response": response,
                "predicted_species": predicted_species,
                "format_compliant": compliant,
            }
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"expected={expected} "
            f"predicted={predicted_species} "
            f"compliant={compliant}"
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    labels = [
        "setosa",
        "versicolor",
        "virginica",
    ]

    precision, recall, _, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    per_class = {}

    for i, species in enumerate(labels):
        per_class[species] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }

    format_compliance = (
        compliant_count / len(df)
        if len(df) > 0
        else 0.0
    )

    return {
        "version": version,
        "num_samples": len(df),
        "accuracy": float(accuracy),
        "format_compliance_rate": float(
            format_compliance
        ),
        "per_class": per_class,
        "predictions": records,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        required=True,
    )

    parser.add_argument(
        "--test-csv",
        required=True,
    )

    parser.add_argument(
        "--endpoint",
        required=True,
    )

    parser.add_argument(
        "--project",
        default=os.getenv(
            "GOOGLE_CLOUD_PROJECT"
        ),
    )

    parser.add_argument(
        "--location",
        default=os.getenv(
            "GOOGLE_CLOUD_LOCATION",
            "us-central1",
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if not args.project:
        raise ValueError(
            "Set GOOGLE_CLOUD_PROJECT "
            "or pass --project."
        )

    print("=" * 60)
    print(f"Evaluating {args.version}")
    print("=" * 60)

    print(f"Test set: {args.test_csv}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Project: {args.project}")
    print(f"Location: {args.location}")

    df = pd.read_csv(args.test_csv)

    required_columns = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    vertexai.init(
        project=args.project,
        location=args.location,
    )

    model = GenerativeModel(args.endpoint)

    results = evaluate_model(
        model=model,
        df=df,
        version=args.version,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Format compliance: "
        f"{results['format_compliance_rate']:.4f}"
    )

    print()
    print("Per-class metrics:")

    for species, metrics in (
        results["per_class"].items()
    ):
        print(
            f"{species:12s} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()