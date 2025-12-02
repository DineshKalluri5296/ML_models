import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df= pd.read_csv("kyphosis.csv")

df.isnull().sum()
df.columns

x=df[['Age', 'Number', 'Start']]
y=df["Kyphosis"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


model=LogisticRegression()

model.fit(x_train,y_train)

joblib.dump(model, "kyphosis.pkl")
print("✅ Model saved as kyphosis.pkl")
