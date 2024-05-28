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

dataset = pd.read_csv("Housing.csv")

X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

le = LabelEncoder()
for i in [4, 5, 6, 7, 8, 10, 11]:
    X[:, i] = le.fit_transform(X[:, i])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1))

regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
regressor.fit(X_train, Y_train)

pred_train = scaler_Y.inverse_transform(regressor.predict(X_train).reshape(-1, 1))

pred_test = scaler_Y.inverse_transform(regressor.predict(X_test).reshape(-1, 1))


plt.figure(figsize=(10, 6))
plt.scatter(scaler_Y.inverse_transform(Y_test), pred_test, color='green', label='Actual vs Predicted (Test)')
plt.plot([min(scaler_Y.inverse_transform(Y_test)), max(scaler_Y.inverse_transform(Y_test))],
         [min(scaler_Y.inverse_transform(Y_test)), max(scaler_Y.inverse_transform(Y_test))],
         color='red', linestyle='--', label='Perfect Prediction')
plt.title('SVR Model: Actual vs Predicted (Test)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
