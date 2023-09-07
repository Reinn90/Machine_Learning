#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:58:10 2023

COMP3032 - Spring 2023 - Machine Learning
@author: Kevin Reyes - 20658133

Assignment 1 - DUE 5:00pm Wednesday, 27th September 2023

"""

# Set current working directory                    - TODO: check it works 
import os 
os.chdir('.') # set to current folder location
os.getcwd()   # print current working directory

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

''' 
Task 1: Blood pressure Analysis 
'''
# Load in the data
df = pd.read_csv('bloodpressure-23.csv', index_col = None)

# View the data and its attributes
df.head()

# Dimensions
df.shape # 100 rows, 12 columns

# Column names
df.columns.values

'''
2. Create polynomial regression models, to predict systolic pressure 
using the SERUM-CHOL feature, for degrees vary from 1 to 14. 
Perform 10-fold cross validation. Calculate its square roots of the 
mean square errors (RMSE), and the mean RMSE.
Display the mean RMSEs for the 14 different degrees. Produce a cross validation
error plot using the mean RMSE with 1 to 14 different degrees
'''

# Set the feature and target variables
X = df[['SERUM-CHOL']]
y = df['SYSTOLIC']

# Perform Polynomial regression model for degrees 1-14
# Conduct a loop for the varying degrees using Pipeline

degrees = list(range(1,15))  # degree list from 1-14
mean_rmse = []               # list to store the mean_RMSE from each model

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), 
                          StandardScaler(), LinearRegression())
    rmse = np.sqrt(-cross_val_score(model, X, y, 
                                    scoring = 'neg_mean_squared_error',
                                    cv = 10))
    mean_rmse.append(np.mean(rmse))
    
# display the mean_rmse
for degree, rmse in zip(degrees, mean_rmse):
    print(f"Degree {degree}: Mean RMSE = {rmse}")

# Create the cross-validation error plot
plt.plot(degrees, mean_rmse, marker='o')
plt.xlabel("Degree")
plt.ylabel("Mean RMSE")
plt.title("Cross-Validation Error Plot")
plt.show()



