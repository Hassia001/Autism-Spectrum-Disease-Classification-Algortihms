# ---------------------------logistic equation-------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import classification_report
data=pd.read_csv('ASD1_data.csv')

len(data.columns)
pd.options.display.max_columns=None
pd.options.display.max_rows=None

print (data.shape)
X=data.drop(['Class/ASD'],axis=1)
Y=data['Class/ASD']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.4, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression ()
logmodel.fit(X_train, Y_train)

y_pred=logmodel.predict(X_test)
(logmodel.score(X_test, Y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y_test, y_pred)
print(confusion_matrix)

print(classification_report(Y_test,y_pred))