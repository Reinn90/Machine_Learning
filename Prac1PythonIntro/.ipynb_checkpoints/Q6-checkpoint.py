#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:29:42 2023

@author: kevinr
"""


# 6. Create pandas dataframe from external source
# such as a csv file using read_csv(<file>) function

import pandas as pd
cars = pd.read_csv('car_data.csv')
print(cars)
