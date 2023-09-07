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



