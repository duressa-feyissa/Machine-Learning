#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:02:39 2024

@author: Duresa Feyisa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


dataset = pd.read_csv("Housing.csv")
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

le = LabelEncoder()
for i in [4, 5, 6, 7, 8, 10, 11]:
    X[:, i] = le.fit_transform(X[:, i])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).ravel()

regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
regressor.fit(X_train, Y_train)

pred_train = scaler_Y.inverse_transform(regressor.predict(X_train).reshape(-1, 1))
pred_test = scaler_Y.inverse_transform(regressor.predict(X_test).reshape(-1, 1))
Y_test_original = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))

mae = mean_absolute_error(Y_test_original, pred_test)
rmse = np.sqrt(mean_squared_error(Y_test_original, pred_test))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

comparison_df = pd.DataFrame({'Actual': Y_test_original.flatten(), 'Predicted': pred_test.flatten()})
print(comparison_df.head(10))

