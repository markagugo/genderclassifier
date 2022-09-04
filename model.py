import imp
from statistics import mode
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('gcd.csv')

data['gender'] = data['gender'].replace({
    'Male': 1,
    'Female': 0
})

x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=48, weights='distance')

knn.fit(x_train, y_train)

y_predict=knn.predict(x_test)

import pickle

pickle.dump(knn, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_predict)


# # Selecting The Best K
# from sklearn.model_selection import cross_val_score

# scores_1 = []
# for i in range(1, 50):
#     knn2 = KNeighborsClassifier(n_neighbors=i, weights='distance')
#     score_2 = cross_val_score(knn2, x, y, cv=10)
#     scores_1.append(score_2.mean())


# plt.figure(figsize=(20, 10))
# plt.plot(range(1, 50), scores_1, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='red' ,markersize='10')
# plt.title('Accuracy Rate vs K-Values')
# plt.xlabel('k')
# plt.ylabel('Accuracy')
# plt.show()

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(solver='lbfgs')
# model.fit(x_train, y_train)

# ans = model.predict(x_test)


# from sklearn.naive_bayes import GaussianNB

# diab_model = GaussianNB()

# diab_model.fit(x_train, y_train.ravel())

# from sklearn import svm

# svm1 = svm.SVC(C=3)

# svm1.fit(x_train, y_train)
# print(svm1.score(x_test, y_test))

# # Selecting The Best C
# from sklearn.model_selection import cross_val_score

# scores_1 = []
# for i in range(1, 50):
#     svm2 = svm.SVC(C=i)
#     score_2 = cross_val_score(svm2, x, y, cv=10)
#     scores_1.append(score_2.mean())


# plt.figure(figsize=(20, 10))
# plt.plot(range(1, 50), scores_1, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='red' ,markersize='10')
# plt.title('Accuracy Rate vs C-Values')
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.show()