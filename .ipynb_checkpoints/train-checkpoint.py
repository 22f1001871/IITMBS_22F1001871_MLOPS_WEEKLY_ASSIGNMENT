#Script for training the model. 

import os
import joblib

#from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

os.mkdirs("data",exist_ok=True)
os.mkdirs("model",exist_ok=True)


data = pd.read_csc()

data.to_csv("data/iris.csv",index=False)

print("Data has been recived from the cloud storage", df.shape())

X = data.drop['species']
y = data['species']

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X,y)

joblib.dump(model,"./model/model.joblib")

print("Iteration 1 is done")