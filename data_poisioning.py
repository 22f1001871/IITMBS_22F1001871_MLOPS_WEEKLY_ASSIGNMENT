import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load the original IRIS dataset
# -------------------------------
df = pd.read_csv("data/iris.csv")

# Rename columns
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target"
]

# Encode string labels to integers
le = LabelEncoder()
df["target"] = le.fit_transform(df["target"])

# Save the clean dataset
df.to_csv("data/iris_clean.csv", index=False)

# -------------------------------
# Poisoning function
# -------------------------------
def poison_dataset(df, corruption_percent, random_state=42):
    poisoned_df = df.copy()

    rng = np.random.default_rng(random_state)

    n_samples = len(poisoned_df)
    n_poison = int(round(n_samples * corruption_percent))

    # Randomly choose rows to poison
    poison_indices = rng.choice(
        poisoned_df.index,
        size=n_poison,
        replace=False
    )

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    # Feature ranges
    mins = df[feature_columns].min()
    maxs = df[feature_columns].max()

    # Replace selected rows
    for idx in poison_indices:

        # Replace all four features with random values
        for feature in feature_columns:
            poisoned_df.loc[idx, feature] = rng.uniform(
                mins[feature],
                maxs[feature]
            )

        # Replace label with a random class (0, 1, or 2)
        poisoned_df.loc[idx, "target"] = rng.integers(0, 3)

    return poisoned_df


# -------------------------------
# Create poisoned datasets
# -------------------------------
poison_levels = {
    "5": 0.05,
    "10": 0.10,
    "50": 0.50
}

for name, pct in poison_levels.items():

    poisoned = poison_dataset(df, pct)

    filename = f"data/iris_poisoned_{name}.csv"

    poisoned.to_csv(filename, index=False)

    print(f"{filename} created")

print("\nAll datasets created successfully!")