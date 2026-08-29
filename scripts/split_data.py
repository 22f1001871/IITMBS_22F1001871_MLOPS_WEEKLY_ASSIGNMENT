import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/iris.csv")
    p.add_argument("--test", default="data/test.csv")
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--validation-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.input)
    required = {"sepal_length","sepal_width","petal_length","petal_width","species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    train_val, test = train_test_split(
        df, test_size=a.test_size, random_state=a.seed, stratify=df["species"]
    )
    train, validation = train_test_split(
        train_val, test_size=a.validation_size,
        random_state=a.seed, stratify=train_val["species"]
    )

    for path, frame in [
        (a.train, train), (a.validation, validation), (a.test, test)
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    print(f"Total={len(df)}, train={len(train)}, validation={len(validation)}, test={len(test)}")

if __name__ == "__main__":
    main()
