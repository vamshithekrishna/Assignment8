# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 22:36:58 2018

@author: vamshi
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import load_boston
from sklearn import metrics


boston = load_boston()
bos = pd.DataFrame(boston.data,columns=['CRIM', 'ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS','RAD', 'TAX','PTRATIO','B','LSTAT'])
bos.info()
boston.target
bos['Price']=boston.target
boston.keys()
boston['feature_names']
X = bos.iloc[:,:-1].values
y= bos.iloc[:,13].values

#Divide train and test the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.29, random_state=1)

#Create a model for Logistic Regreassion
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)










