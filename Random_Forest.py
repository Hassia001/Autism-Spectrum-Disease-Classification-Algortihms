import pandas as pd
import numpy as np

data = pd.read_csv('ASD1_data.csv')

# X = data.drop(axis=1, labels='Class/ASD')
X = data.drop('Class/ASD', axis=1)
y = data['Class/ASD']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# training algorithm with 20 trees
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
