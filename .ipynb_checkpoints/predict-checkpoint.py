import joblib
from feast import FeatureStore
import pandas as pd

data = pd.read_csv("feature_repo/data/iris_data_adapted_for_feast.csv")
data["event_timestamp"] = pd.to_datetime(data["event_timestamp"])

model = joblib.load("artifacts/model.joblib")


store = FeatureStore(repo_path = "feature_repo")

print("\n\n")
iris_id = int(input("Enter the IRIS ID : "))

online_features = store.get_online_features(
    features = [
        "iris_features:sepal_length",
         "iris_features:sepal_width",
         "iris_features:petal_length",
         "iris_features:petal_width",
    ],
    entity_rows = [
        {"iris_id":iris_id},
    ],
).to_df()

X =  online_features[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]
if online_features.isnull().any().any():
    print(f"No features found for iris_id {iris_id}")
    exit()

predictions = model.predict(X)

print("\n\n")
print("The predicted species are")
print("------------------------------------")
print(predictions[0])

#comparing predictions with the raw data
raw = (data[data["iris_id"]==iris_id].sort_values("event_timestamp").iloc[-1])
raw_features = raw.drop(labels=["iris_id", "species", "event_timestamp", "created_timestamp"])

X = pd.DataFrame([raw_features])

raw_prediction = model.predict(X)

print("\n\n")
print("The actual raw dataset prediction")
print("------------------------------------------")
print( raw_prediction[0])