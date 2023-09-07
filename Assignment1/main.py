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

# Load in the data
df = pd.read_csv('bloodpressure-23.csv', index_col = None)

# View the data and its attributes
df.head()

# Dimensions
df.shape # 100 rows, 12 columns

# Column names
df.columns.values



