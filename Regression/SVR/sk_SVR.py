#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:27:22 2017

@author: sparshkoyarala
"""

# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

# Visualising the SVR Result

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('SVR Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Visualising in Higher Def, the SVR Result

X_grid = np.arange(min(X), max(X), 0.1)
X_grid  = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('SVR Regression - High Res')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

