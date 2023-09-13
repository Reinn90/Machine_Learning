#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:29:42 2023

@author: kevinr
"""

# 5. Creating Pandas DataFrames
import pandas as pd

d = {'make' : ['Ford','Toyota','Hyundai','Honda','Subaru'],
        'model' : ['Everest','Kluger','Santa Fe','CR-V','Forrester'],
        'year' : [2016,2005,2010,2018,2017],
        'fuel' : ['Diesel','Petrol','Diesel','Petrol','Petrol']
        }

df = pd.DataFrame(d)
print(df)
print('\nCars Dataset')

W