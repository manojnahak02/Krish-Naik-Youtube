# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:08:44 2019

@author: Manoj Nahak
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1] #except the last column, all other columns have been taken
y = dataset.iloc[:, 4] #only the last column is been taken

#Convert the categorical columns through one-hot-encoding (because of 3 levels)

states=pd.get_dummies(x['State'],drop_first=True)#dummy variable created throught get_dummies 

# Drop the state coulmn
x=x.drop('State',axis=1)

# concat the dummy variables
x=pd.concat([x,states],axis=1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the output Test set results
y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
#R2 value is = 0.9347 (close to 1, means model is good)

