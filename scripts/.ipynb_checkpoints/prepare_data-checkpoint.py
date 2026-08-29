import argparse
import json
from pathlib import Path
import pandas as pd

def make_record(row, version):
    sl = row["sepal_length"]
    sw = row["sepal_width"]
    pl = row["petal_length"]
    pw = row["petal_width"]

    species = str(row["species"]).strip().lower()

    if version == "v1":
        input_text = (
            f"sepal_length: {sl}, "
            f"sepal_width: {sw}, "
            f"petal_length: {pl}, "
            f"petal_width: {pw}"
        )

        output_text = species

    else:
        input_text = (
            f"A flower specimen has a sepal length of {sl} cm, "
            f"sepal width of {sw} cm, "
            f"petal length of {pl} cm, "
            f"and petal width of {pw} cm. "
            f"Identify the iris species."
        )

        output_text = f"This is Iris {species}."

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": input_text}
                ],
            },
            {
                "role": "model",
                "parts": [
                    {"text": output_text}
                ],
            },
        ]
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--version", choices=["v1","v2"], required=True)
    a = p.parse_args()

    df = pd.read_csv(a.input)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)

    with open(a.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(make_record(row, a.version)) + "\n")

    print(f"Created {a.output}: {len(df)} records")

if __name__ == "__main__":
    main()
