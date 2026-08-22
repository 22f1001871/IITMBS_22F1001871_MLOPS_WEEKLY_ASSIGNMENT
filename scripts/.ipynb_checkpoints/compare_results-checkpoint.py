import argparse
import json
from pathlib import Path


def load_results(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1",
        default="results/v1_results.json",
    )

    parser.add_argument(
        "--v2",
        default="results/v2_results.json",
    )

    parser.add_argument(
        "--output",
        default="results/comparison.json",
    )

    args = parser.parse_args()

    v1 = load_results(args.v1)
    v2 = load_results(args.v2)

    comparison = {
        "v1": {
            "accuracy": v1["accuracy"],
            "format_compliance_rate": (
                v1["format_compliance_rate"]
            ),
            "per_class": v1["per_class"],
        },
        "v2": {
            "accuracy": v2["accuracy"],
            "format_compliance_rate": (
                v2["format_compliance_rate"]
            ),
            "per_class": v2["per_class"],
        },
        "difference_v2_minus_v1": {
            "accuracy": (
                v2["accuracy"] -
                v1["accuracy"]
            ),
            "format_compliance_rate": (
                v2["format_compliance_rate"] -
                v1["format_compliance_rate"]
            ),
        },
    }

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
            comparison,
            f,
            indent=2,
        )

    print("=" * 70)
    print("V1 vs V2")
    print("=" * 70)

    print(
        f"{'Metric':<30}"
        f"{'V1':>12}"
        f"{'V2':>12}"
    )

    print("-" * 70)

    print(
        f"{'Accuracy':<30}"
        f"{v1['accuracy']:>12.4f}"
        f"{v2['accuracy']:>12.4f}"
    )

    print(
        f"{'Format compliance':<30}"
        f"{v1['format_compliance_rate']:>12.4f}"
        f"{v2['format_compliance_rate']:>12.4f}"
    )

    print()
    print("Per-class metrics")
    print("-" * 70)

    for species in [
        "setosa",
        "versicolor",
        "virginica",
    ]:

        v1_metrics = v1["per_class"][species]
        v2_metrics = v2["per_class"][species]

        print(f"\n{species}")

        print(
            f"  Precision: "
            f"{v1_metrics['precision']:.4f} "
            f"→ "
            f"{v2_metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{v1_metrics['recall']:.4f} "
            f"→ "
            f"{v2_metrics['recall']:.4f}"
        )

    print()
    print("=" * 70)

    if v1["accuracy"] > v2["accuracy"]:
        print("Higher accuracy: V1")
    elif v2["accuracy"] > v1["accuracy"]:
        print("Higher accuracy: V2")
    else:
        print("Accuracy: TIE")

    if (
        v1["format_compliance_rate"]
        >
        v2["format_compliance_rate"]
    ):
        print("Higher format compliance: V1")
    elif (
        v2["format_compliance_rate"]
        >
        v1["format_compliance_rate"]
    ):
        print("Higher format compliance: V2")
    else:
        print("Format compliance: TIE")

    print("=" * 70)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()