# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:36:52 2022

@author: Dell
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-7].values


Y1 = dataset.iloc[:,32:33].values
Y2 = dataset.iloc[:,33:34].values
Y3 = dataset.iloc[:,34:35].values
Y4 = dataset.iloc[:,35:36].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:, 1:])



# Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 0)
X_train, X_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 0)
X_train, X_test, Y3_train, Y3_test = train_test_split(X, Y3, test_size = 0.2, random_state = 0)
X_train, X_test, Y4_train, Y4_test = train_test_split(X, Y4, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()


#Predicting Test set results of Y1
regressor.fit(X_train,Y1_train)
y1_pred = regressor.predict(X_test)

#Predicting Test set results of Y2
regressor.fit(X_train,Y2_train)
y2_pred = regressor.predict(X_test)

#Predicting Test set results of Y3
regressor.fit(X_train,Y3_train)
y3_pred = regressor.predict(X_test)

#Predicting Test set results of Y4
regressor.fit(X_train,Y4_train)
y4_pred = regressor.predict(X_test)



#Evaluate Model
from sklearn.metrics import r2_score


print("For Y1 accuracy is:")
print(r2_score(Y1_test,y1_pred),"\n")
Y1_test = Y1_test.flatten()
import matplotlib.pyplot as plt
plt.scatter(y1_pred,(Y1_test-y1_pred),marker='^')
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.show()

print("For Y2 accuracy is:")
print(r2_score(Y2_test,y2_pred),"\n")
Y2_test = Y2_test.flatten()
import matplotlib.pyplot as plt
plt.scatter(y2_pred,(Y2_test-y2_pred),marker='^')
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.show()

print("For Y3 accuracy is:")
print(r2_score(Y3_test,y3_pred),"\n")
Y3_test = Y3_test.flatten()
import matplotlib.pyplot as plt
plt.scatter(y3_pred,(Y3_test-y3_pred),marker='^')
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.show()

print("For Y4 accuracy is:")
print(r2_score(Y4_test,y4_pred),"\n")
Y4_test = Y4_test.flatten()
import matplotlib.pyplot as plt
plt.scatter(y4_pred,(Y4_test-y4_pred),marker='^')
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.show()



























