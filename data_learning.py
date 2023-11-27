# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:07:11 2023

@author: lorena
"""

#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

#%%
import pickle

with open('data_preprocessed.pkl', 'rb') as file:
    previsores_treino, previsores_teste, classes_treino, classes_teste = pickle.load(file)
    
#%%
""" Decision Tree Classifier"""
from sklearn.tree import DecisionTreeClassifier
tree_census_data = DecisionTreeClassifier(criterion='entropy', random_state= 0)
tree_census_data.fit(previsores_treino, classes_treino)

previsoes = tree_census_data.predict(previsores_teste)

print(accuracy_score(classes_teste, previsoes))
# 0.9885802469135803

print(confusion_matrix(classes_teste, previsoes)) 
# [[ 3263,   322],
#  [  307, 51188]]

print(classification_report(classes_teste, previsoes))
#               precision    recall  f1-score   support
# 
#      ANORMAL       0.91      0.91      0.91      3585
#       NORMAL       0.99      0.99      0.99     51495
# 
#     accuracy                           0.99     55080
#    macro avg       0.95      0.95      0.95     55080
# weighted avg       0.99      0.99      0.99     55080

#%%
""" Naive Bayes Classifier """
from sklearn.naive_bayes import GaussianNB
naive_census_data = GaussianNB()
naive_census_data.fit(previsores_treino, classes_treino)

previsoes = naive_census_data.predict(previsores_teste)

print(accuracy_score(classes_teste, previsoes))
# 0.9750544662309368

print(confusion_matrix(classes_teste, previsoes))
# [[ 3491    94]
#  [ 1280 50215]]

print(classification_report(classes_teste, previsoes))
#               precision    recall  f1-score   support
#
#      ANORMAL       0.73      0.97      0.84      3585
#       NORMAL       1.00      0.98      0.99     51495
#
#     accuracy                           0.98     55080
#    macro avg       0.86      0.97      0.91     55080
# weighted avg       0.98      0.98      0.98     55080

#%%
""" K-Nearest-Neighbors Classifier """
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classificador.fit(previsores_treino, classes_treino)
previsoes = classificador.predict(previsores_teste)

print(accuracy_score(classes_teste, previsoes))
# 0.985039941902687

print(confusion_matrix(classes_teste, previsoes))
# [[ 2964   621]
#  [  203 51292]]

print(classification_report(classes_teste, previsoes))
#               precision    recall  f1-score   support
#
#      ANORMAL       0.94      0.83      0.88      3585
#       NORMAL       0.99      1.00      0.99     51495

#     accuracy                           0.99     55080
#    macro avg       0.96      0.91      0.93     55080
# weighted avg       0.98      0.99      0.98     55080

#%%
""" Logistic Regression Classifier """
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state = 1)
classificador.fit(previsores_treino, classes_treino)
previsoes = classificador.predict(previsores_teste)

print(accuracy_score(classes_teste, previsoes))
# 0.9911401597676107

print(confusion_matrix(classes_teste, previsoes))
# [[ 3344   241]
#  [  247 51248]]

print(classification_report(classes_teste, previsoes))
#               precision    recall  f1-score   support
#
#      ANORMAL       0.93      0.93      0.93      3585
#       NORMAL       1.00      1.00      1.00     51495
#
#     accuracy                           0.99     55080
#    macro avg       0.96      0.96      0.96     55080
# weighted avg       0.99      0.99      0.99     55080

#%%
from sklearn.svm import SVC
classificador = SVC(random_state = 1, C = 1.0 , kernel = 'linear')
classificador.fit(previsores_treino, classes_treino)
previsoes = classificador.predict(previsores_teste)

print(accuracy_score(classes_teste, previsoes))
# 0.9920116194625999

print(confusion_matrix(classes_teste, previsoes))
# [[ 3430   155]
#  [  285 51210]]

print(classification_report(classes_teste, previsoes))
#               precision    recall  f1-score   support
#
#      ANORMAL       0.92      0.96      0.94      3585
#       NORMAL       1.00      0.99      1.00     51495
# 
#     accuracy                           0.99     55080
#    macro avg       0.96      0.98      0.97     55080
# weighted avg       0.99      0.99      0.99     55080

#%%


