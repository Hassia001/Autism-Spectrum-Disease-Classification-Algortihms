import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


dataset = pd.read_csv("ASD1_data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 30].values

knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
print(cv_scores)
mean = np.mean(cv_scores)
print (mean)

dataset1 = dataset.to_numpy()
clf = KNeighborsClassifier()

n_feats = dataset.shape[1]

print('Feature  Accuracy')
for i in range(n_feats):
    X = dataset1[:, i].reshape(-1, 1)
    scores = cross_val_score(clf, X, y, cv=10)
    print('%d        %g' % (i, scores.mean()))
