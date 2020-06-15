#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:50:24 2020

@author: ironman
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Life Expectancy Data.csv')
X = dataset.iloc[:,:].values
X = np.delete(X,3,axis=1)
Y = dataset.iloc[:,3:4].values

#Taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = imputer.fit(X[:, 3:23])
X[:, 3:23] = imputer.transform(X[:, 3:23])
imputer_1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer_1 = imputer_1.fit(Y[:,0:1])
Y[:,0:1] = imputer_1.transform(Y[:,0:1])

#Label Encoding and OneHotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Labelencoder_X_1 = LabelEncoder()
X[:,0] = Labelencoder_X_1.fit_transform(X[:,0])
Labelencoder_X_2 = LabelEncoder()
X[:,2] = Labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting train and test cases
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators = 20)
Regressor.fit(X_train,Y_train)
y_pred = Regressor.predict(X_test)

#Evaluating the score of train results
score_train = Regressor.score(X_train,Y_train)

#Evaluating the score of test results
score_test = Regressor.score(X_test,Y_test)

#printing the values of the scores
print(score_train)
print(score_test)




