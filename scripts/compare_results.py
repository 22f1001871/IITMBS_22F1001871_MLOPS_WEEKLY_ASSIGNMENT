import argparse
import json


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1",
        required=True,
        help="Path to V1 evaluation results JSON"
    )

    parser.add_argument(
        "--v2",
        required=True,
        help="Path to V2 evaluation results JSON"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output comparison JSON"
    )

    args = parser.parse_args()

    v1 = load_results(args.v1)
    v2 = load_results(args.v2)

    # ---------------------------------------------------------
    # Support both old and new evaluate.py JSON formats
    # ---------------------------------------------------------

    if "metrics" in v1:
        m1 = v1["metrics"]
    else:
        m1 = v1

    if "metrics" in v2:
        m2 = v2["metrics"]
    else:
        m2 = v2

    v1_accuracy = m1["accuracy"]
    v2_accuracy = m2["accuracy"]

    v1_compliance = m1["format_compliance"]
    v2_compliance = m2["format_compliance"]

    v1_per_class = m1["per_class"]
    v2_per_class = m2["per_class"]

    # ---------------------------------------------------------
    # Build comparison
    # ---------------------------------------------------------

    comparison = {
        "v1": {
            "accuracy": v1_accuracy,
            "format_compliance": v1_compliance,
            "per_class": v1_per_class,
        },

        "v2": {
            "accuracy": v2_accuracy,
            "format_compliance": v2_compliance,
            "per_class": v2_per_class,
        },

        "difference": {
            "accuracy": v2_accuracy - v1_accuracy,
            "format_compliance": v2_compliance - v1_compliance,
        },

        "winner": {
            "accuracy": (
                "v1"
                if v1_accuracy > v2_accuracy
                else "v2"
                if v2_accuracy > v1_accuracy
                else "tie"
            ),

            "format_compliance": (
                "v1"
                if v1_compliance > v2_compliance
                else "v2"
                if v2_compliance > v1_compliance
                else "tie"
            ),
        },
    }

    # ---------------------------------------------------------
    # Print comparison
    # ---------------------------------------------------------

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print()

    print("                    V1          V2          Difference")
    print("-" * 60)

    print(
        f"Accuracy            "
        f"{v1_accuracy:.4f}      "
        f"{v2_accuracy:.4f}      "
        f"{v2_accuracy - v1_accuracy:+.4f}"
    )

    print(
        f"Format compliance   "
        f"{v1_compliance:.4f}      "
        f"{v2_compliance:.4f}      "
        f"{v2_compliance - v1_compliance:+.4f}"
    )

    print()
    print("=" * 60)
    print("PER-CLASS COMPARISON")
    print("=" * 60)

    for species in [
        "setosa",
        "versicolor",
        "virginica",
    ]:

        v1_precision = v1_per_class[species]["precision"]
        v1_recall = v1_per_class[species]["recall"]

        v2_precision = v2_per_class[species]["precision"]
        v2_recall = v2_per_class[species]["recall"]

        print()
        print(species)

        print(
            f"  Precision: "
            f"V1={v1_precision:.4f} "
            f"V2={v2_precision:.4f}"
        )

        print(
            f"  Recall:    "
            f"V1={v1_recall:.4f} "
            f"V2={v2_recall:.4f}"
        )

    print()
    print("=" * 60)

    print(
        f"Accuracy winner: "
        f"{comparison['winner']['accuracy']}"
    )

    print(
        f"Format compliance winner: "
        f"{comparison['winner']['format_compliance']}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2
        )

    print()
    print(f"Saved comparison to: {args.output}")


if __name__ == "__main__":
    main()