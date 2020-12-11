#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
data = pd.read_csv('ASD1_data.csv')
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
X = data.drop('Class/ASD', axis=1)
y = data['Class/ASD']
# print(data)
#print(data.iloc[0:13,0:13])

# randomly split the data into training and testing sets.
# 20% data for test, 80% data for train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# train the algorithm on the training data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# make predictions on the test data
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

