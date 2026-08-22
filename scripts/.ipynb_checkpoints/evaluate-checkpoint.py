import argparse
import json
import os
import re

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel


VALID_SPECIES = {"setosa", "versicolor", "virginica"}


def extract_species(response: str):
    """
    Extract a valid Iris species from the model response.

    This is used for accuracy/precision/recall.

    Format compliance is calculated separately and strictly:
    the complete response must be exactly one of:
        setosa
        versicolor
        virginica
    """

    if not response:
        return None

    text = response.strip().lower()

    # Exact valid output
    if text in VALID_SPECIES:
        return text

    # Look for species anywhere in the response.
    # This allows us to calculate classification accuracy even when
    # the model adds extra text.
    for species in VALID_SPECIES:
        if re.search(rf"\b{species}\b", text):
            return species

    return None


def is_format_compliant(response: str):
    """
    Strict format compliance.

    A response is compliant ONLY when the entire response is exactly:
        setosa
        versicolor
        virginica

    Examples:

        "setosa"                  -> True
        "versicolor"              -> True
        "Setosa"                  -> False
        "This is Iris setosa."    -> False
        '{"species": "setosa"}'   -> False
        "setosa."                 -> False
        ""                        -> False
    """

    if not response:
        return False

    return response.strip() in VALID_SPECIES


def calculate_metrics(results):
    """
    Calculate accuracy, per-class precision and recall,
    and format compliance.
    """

    total = len(results)

    correct = sum(
        1
        for r in results
        if r["predicted"] == r["expected"]
    )

    accuracy = correct / total if total else 0.0

    compliant = sum(
        1
        for r in results
        if r["format_compliant"]
    )

    format_compliance = compliant / total if total else 0.0

    metrics = {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "per_class": {},
    }

    for species in ["setosa", "versicolor", "virginica"]:

        true_positive = sum(
            1
            for r in results
            if r["expected"] == species
            and r["predicted"] == species
        )

        false_positive = sum(
            1
            for r in results
            if r["expected"] != species
            and r["predicted"] == species
        )

        false_negative = sum(
            1
            for r in results
            if r["expected"] == species
            and r["predicted"] != species
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )

        metrics["per_class"][species] = {
            "precision": precision,
            "recall": recall,
        }

    return metrics


def create_input_text(row):
    """
    Create the same raw-feature representation used for evaluation.

    The model was trained using:

    sepal_length: 5.1, sepal_width: 3.5,
    petal_length: 1.4, petal_width: 0.2
    """

    return (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )


def create_prompt(row, version):
    """
    Create the inference prompt.

    Both versions are evaluated using their corresponding
    representation.
    """

    if version == "v1":
        return (
            f"sepal_length: {row['sepal_length']}, "
            f"sepal_width: {row['sepal_width']}, "
            f"petal_length: {row['petal_length']}, "
            f"petal_width: {row['petal_width']}"
        )

    elif version == "v2":
        return (
            f"A flower specimen has a sepal length of "
            f"{row['sepal_length']} cm, sepal width of "
            f"{row['sepal_width']} cm, petal length of "
            f"{row['petal_length']} cm, and petal width of "
            f"{row['petal_width']} cm. "
            f"Identify the iris species."
        )

    raise ValueError(f"Unknown version: {version}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Gemini Iris model."
    )

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
        help="Full Vertex AI endpoint resource name.",
    )

    parser.add_argument(
        "--project",
        required=True,
    )

    parser.add_argument(
        "--location",
        default="us-central1",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Evaluating {args.version}")
    print("=" * 60)

    print(f"Test set: {args.test_csv}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Project: {args.project}")
    print(f"Location: {args.location}")

    # ---------------------------------------------------------
    # Vertex AI initialization
    # ---------------------------------------------------------

    vertexai.init(
        project=args.project,
        location=args.location,
    )

    # Extract endpoint ID from:
    #
    # projects/.../locations/.../endpoints/123456
    #
    endpoint_id = args.endpoint.split("/")[-1]

    model = GenerativeModel(
        args.endpoint
    )

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

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
            f"Missing required columns: {sorted(missing)}"
        )

    results = []

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    for index, row in df.iterrows():

        expected = str(row["species"]).strip().lower()

        prompt = create_prompt(
            row,
            args.version,
        )

        try:

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 500,
                },
            )

            raw_response = ""

            if response is not None:
                try:
                    raw_response = response.text.strip()
                except Exception:
                    raw_response = ""

            # -------------------------------------------------
            # Strict format compliance
            # -------------------------------------------------

            compliant = is_format_compliant(
                raw_response
            )

            # -------------------------------------------------
            # Extract species for classification metrics
            # -------------------------------------------------

            predicted = extract_species(
                raw_response
            )

        except Exception as e:

            print(
                f"[{index + 1}/{len(df)}] "
                f"ERROR: {e}"
            )

            raw_response = ""
            predicted = None
            compliant = False

        results.append(
            {
                "index": int(index),
                "expected": expected,
                "predicted": predicted,
                "raw_response": raw_response,
                "format_compliant": compliant,
            }
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"expected={expected} "
            f"predicted={predicted} "
            f"compliant={compliant}"
        )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(results)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Format compliance: "
        f"{metrics['format_compliance']:.4f}"
    )

    print()
    print("Per-class metrics:")

    for species in [
        "setosa",
        "versicolor",
        "virginica",
    ]:

        precision = metrics["per_class"][
            species
        ]["precision"]

        recall = metrics["per_class"][
            species
        ]["recall"]

        print(
            f"{species:<12} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output = {
        "version": args.version,
        "project": args.project,
        "location": args.location,
        "endpoint": args.endpoint,
        "endpoint_id": endpoint_id,
        "test_set": args.test_csv,
        "num_samples": len(results),
        "metrics": metrics,
        "predictions": results,
    }

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()