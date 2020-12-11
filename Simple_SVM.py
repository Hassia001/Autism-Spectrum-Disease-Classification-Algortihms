import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
data = pd.read_csv('ASD1_data.csv')

# X = data.drop(axis=1, labels='Class/ASD')
X = data.drop('Class/ASD', axis=1)
y = data['Class/ASD']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Training the Algorithm
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', C=0.01)
svclassifier.fit(X_train, y_train)

# Making Predictions
y_pred = svclassifier.predict(X_test)

# Evaluating  the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


