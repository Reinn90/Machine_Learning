#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning 
Module 07: nonparametric methods 
"""

#%% Preamble: packages 
import numpy as np
import numpy as np
import os
import pandas as pd

from sklearn import datasets

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

#%% ########### Section 1: KNN classification ###########
# to make the code output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# kNN classification

# set up iris data
# use sepal width, petal width to predict species
iris = datasets.load_iris()
X = iris["data"][:, (1, 3)]  # sepal width, petal width
y = iris["target"]

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b*", label="Setosa")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Versicolor")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "ro", label="Virginica")

    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel("Sepal width", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center right", fontsize=14)

# Just plot the data    
plt.figure(figsize=(7,6))
plot_dataset(X, y, [1.9, 4.5, 0.0, 2.6])
plt.show()

# Plot the decision boundary (class regions) 
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

k = 1
knn_clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)

plt.figure(figsize=(7, 6))
plot_predictions(knn_clf, [1.9, 4.5, 0.0, 2.6])
plot_dataset(X, y, [1.9, 4.5, 0.0, 2.6])
plt.title("{} Neighbour".format(k),  fontsize=14)
plt.show()


#%% ########### Section 2: KNN regression (kernel smoothing) ###########
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T, y_, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()

#%% ########### Section 3: Use KNN to predict hand written digits ###########
# Common imports
import numpy as np
import os
import pandas as pd

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#---  Collecting all the imports used in this tutorial here: 
# kNN stuff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

# PCA
from sklearn.decomposition import PCA

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# other
import timeit
import tracemalloc

mtrain = pd.read_csv("mnist_train.csv", header=0)
mtest = pd.read_csv("mnist_test.csv", header=0)

print('size of training data: ', mtrain.shape)
print('size of test data: ', mtest.shape)

digit1 = 3
digit2 = 8
mtrain2 = mtrain.loc[(mtrain['label'] == digit1) | (mtrain['label'] == digit2)]
mtest2 = mtest.loc[ (mtest['label'] == digit1) | (mtest['label'] == digit2) ]

# set up the data: 
X = mtrain2.drop(mtrain2.columns[0], axis=1)
y = (mtrain2['label'] == digit2).astype(np.float64)   

# Take the first image and plot it
idx = 0 # You can change it to see others
img = np.c_[X.iloc[idx,:]].reshape(28,28)
plt.imshow(img, cmap="Greys")
plt.title(f'sample label is {y.iloc[idx].astype(np.int64)}')
plt.show()
#%% Section 3: (a) 
# =============================================================================
# (a) Create subsets of 2 digits from both the training and test data 
# (pick two digits, e.g. 3 and 8). Classify the test data using the training set, 
# with k = 3.
# =============================================================================
# Let's just pick "3" and "8" 
digit1 = 3
digit2 = 8
mtrain2 = mtrain.loc[(mtrain['label'] == digit1) | (mtrain['label'] == digit2)]
mtest2 = mtest.loc[ (mtest['label'] == digit1) | (mtest['label'] == digit2) ]

# set up the data: 
X = mtrain2.drop(mtrain2.columns[0], axis=1)
y = (mtrain2['label'] == digit2).astype(np.float64)   

# set up test data:
X_test = mtest2.drop(mtest2.columns[0], axis=1)
y_test = (mtest2['label'] == digit2).astype(np.float64)

# Note: tracemalloc is used here to check memory usage

start = timeit.default_timer()
tracemalloc.start()
knn_clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
y_pred = knn_clf.predict(X_test)

print('Time to run the process: ', timeit.default_timer() - start)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()

# classification accuracy
confusion_matrix(y_pred, y_test)
print(accuracy_score(y_test, y_pred)*100)

#%% Section 3: (b) 
# =============================================================================
# (b) Pick a misclassified example. How do the nearest neighbours look like?
# =============================================================================
indMisclassification = np.where((y_test == y_pred) == False)
indMisclassification[0]
# Pick a misclassified sample and find its neighbours
neigh_dist, neigh_ind = knn_clf.kneighbors(np.c_[X_test.iloc[109,:]].reshape(1,-1), 3, return_distance=True)
neigh_ind

# the misclassified data
labels =[digit1, digit2]
print("The misclassified data has the label: ", labels[(y_test.iloc[109]).astype(np.int64)])
img = np.c_[X_test.iloc[109,:]].reshape(28,28)
plt.imshow(img, cmap="Greys")
plt.show()

# The neighbouring points: 
plt.figure(figsize=(12,4))
for idx, i in np.ndenumerate(neigh_ind[0]):
    plt.subplot(1, 3, idx[0]+1)
    img = np.c_[X.iloc[i,:]].reshape(28,28)
    plt.imshow(img, cmap="Greys")
    plt.title(f'sample label is {labels[(y.iloc[i]).astype(np.int64)]}')

plt.show()


#%% Section 3: (c) 
# =============================================================================
# Use the training data to create training and validation sets. Use the validation 
# set to find the best value for k. Use the k you selected, compute the error on 
# the test set. How does it compare with results from part (a)? 
# =============================================================================
X_train, X_validate, y_train, y_validate = train_test_split(X, y, 
                                                            test_size=0.5, 
                                                            random_state=42)

k_values = np.arange(1, 12, 1)
results_acc = []

for k in k_values:
    print(f'k = {k}')
    start = timeit.default_timer()
    knn_clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn_clf.predict(X_validate)
    print(f'   Time to run the classification:  {round(timeit.default_timer() - start, 3)} seconds')
    # calculate accuracy
    acc = accuracy_score(y_validate, y_pred)*100
    results_acc.append(acc)

plt.figure
plt.plot(k_values, results_acc, '-*')
plt.show()

# What value of k is the best? 
#%% Section 3: (d) 
# =============================================================================
# Apply the selected k to classify all digits of the test set. 
# =============================================================================
# Classify all digits of the test set
tracemalloc.start()
start = timeit.default_timer()

# set up the data: 
X = mtrain.drop(mtrain.columns[0], axis=1)
y = (mtrain['label']).astype(np.float64)  

X_test = mtest2.drop(mtest.columns[0], axis=1)
y_test = (mtest2['label']).astype(np.float64)

knn_clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
y_pred = knn_clf.predict(X_test)

print(f'Time to run the process:  {round(timeit.default_timer() - start, 3)} seconds')
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()

print(f'Accuracy on the test data using the new subset is: {round(accuracy_score(y_test, y_pred)*100,2)} %')


#%% ########### Section ?: Use KNN to predict diabetes ###########
# We use sklearn data
# First 10 columns are numeric predictive values
# The last one is blood sugar level we predict. 
# See the following website for the data description
# https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
# Partial code available below. 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = neighbors.KNeighborsRegressor(1, weights='distance')

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
# print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.scatter(diabetes_X_test[:,0], diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
