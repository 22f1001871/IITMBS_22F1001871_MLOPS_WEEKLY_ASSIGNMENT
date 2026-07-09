#Script to train the model on a dataset that is avialble in the local storage. 
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import joblib

data = pd.read_csv("data/iris.csv")

print("Info of the dataset")
print("-------------------")
data.info()
print("-------------------")

X = data.drop(columns=['species'])
y = data['species']

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X,y)

joblib.dump(model, 'models/model.joblib')

print("Done!!")