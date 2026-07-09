import pandas as pd
import pytest

DATA_PATH = "data/iris.csv"

EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

NUMERIC_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_dataset_exists(df):
    """Check dataset is not empty."""
    assert not df.empty


def test_expected_schema(df):
    """Check expected columns exist."""
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values(df):
    """Check there are no missing values."""
    assert df.isnull().sum().sum() == 0


def test_numeric_feature_types(df):
    """Check feature columns are numeric."""
    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_reasonable_value_ranges(df):
    """Check feature values fall within expected Iris ranges."""
    assert df["sepal_length"].between(4.0, 8.0).all()
    assert df["sepal_width"].between(2.0, 5.0).all()
    assert df["petal_length"].between(1.0, 7.0).all()
    assert df["petal_width"].between(0.0, 3.0).all()


def test_target_values(df):
    """Check target labels are valid."""
    valid_labels = {
        "setosa",
        "versicolor",
        "virginica"
    }

    assert set(df["species"]).issubset(valid_labels)