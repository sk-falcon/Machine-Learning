#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:50:07 2017

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

# Fitting the Random Forest Regression

from sklearn.ensemble import RandomForestRegressor



# Predicting at 6.5

y_pred = regressor.predict(6.5)

# Visualising in Higher Def, the SVR Result

X_grid = np.arange(min(X), max(X), 0.01)
X_grid  = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression - High Res')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

