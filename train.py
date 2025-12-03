import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score
# collecting the data from various sources
USA_housing=pd.read_csv("/content/USA_Housing.csv")

#  data preprocessing

USA_housing.head()

USA_housing.info()

USA_housing.describe()

USA_housing.isnull().sum()
print(USA_housing.columns)


# Exploratory Data Analysis:

sns.pairplot(USA_housing)

# Train_test_split

x=USA_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=USA_housing["Price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

lm= LinearRegression() # step2: creating the object

lm.fit(x_train,y_train) # step3: fit

print(lm.intercept_)
predictions=lm.predict(x_test) # step4: predict
plt.scatter(y_test,predictions)
r2_score=r2_score(y_test,predictions)
print(r2_score)

joblib.dump(model, "USA_Housing.pkl")
print("✅ Model saved as USA_Housing.pkl")
