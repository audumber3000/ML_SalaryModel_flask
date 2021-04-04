import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib


# Importing the dataset
dataset = pd.read_csv('/home/audumber/Desktop/BasicMLModelFlaskDeploy/dataset/Sales_Salary_Data.csv')

# seprate feature & target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving serialized model to disk

pickle.dump(regressor, open('/home/audumber/Desktop/BasicMLModelFlaskDeploy/model.pkl','wb'))
#joblib.dump(regressor, 'model.pkl')


# Loading model to compare the results
model = pickle.load(open('/home/audumber/Desktop/BasicMLModelFlaskDeploy/model.pkl','rb'))
#model = joblib.load('model.pkl')

print("Regressor model output", regressor.predict([[1.8]]))
print("Saved  model output", model.predict([[1.8]]))

