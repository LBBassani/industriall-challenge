# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:38:50 2023

@author: lorena
"""


#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import pandas as pd
import numpy as np

#%%
print('Testando os modelos nas três primeiras anomalias da base de dados original')
print('\tLendo os dados do arquivo data/preprocessed_base.csv')
base = pd.read_csv('data/preprocessed_base.csv')

print('\tLendo os modelos treinados do arquivo data_classifiers.pkl')
with open('data_classifiers.pkl', mode='rb') as file:
    tree_classifier, naive_classifier, knn_classifier, logistic_classifier, svc_classifier, neural_network_classifier = pickle.load(file)

print('\tLendo o normalizador dos dados de entrada do arquivo data_scaler.pkl')
with open('data_scaler.pkl', mode='rb') as file:
    scaler = pickle.load(file)

#%%
""" Testing ANORMAL occurrences """
print('\tSeparando as três primeiras ocorrencias de anomalias')
# From 17000 to 18999
anormal_occurance_area_1 = base[17000:19000]
previsores_teste_1 = scaler.transform(anormal_occurance_area_1.iloc[:, 2:])
classes_teste_1 = anormal_occurance_area_1.iloc[:, 1:2]

# From 24000 to 27999
anormal_occurance_area_2 = base[24000:28000]
previsores_teste_2 = scaler.transform(anormal_occurance_area_2.iloc[:, 2:])
classes_teste_2 = anormal_occurance_area_2.iloc[:, 1:2]

# From 127000 to 133999
anormal_occurance_area_3 = base[127000:140000]
previsores_teste_3 = scaler.transform(anormal_occurance_area_3.iloc[:, 2:])
classes_teste_3 = anormal_occurance_area_3.iloc[:, 1:2]

previsores_teste = np.concatenate([previsores_teste_1, previsores_teste_2, previsores_teste_3])
classes_teste = np.concatenate([classes_teste_1, classes_teste_2, classes_teste_3])

#%%
""" Decision Tree Classifier """
print('Testando o Decision Tree Classifier')
previsoes_tree_classifier = tree_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_tree_classifier)}')
# 0.9831578947368421

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_tree_classifier)}') 
# [[12232   215]
#  [  105  6448]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_tree_classifier)}')
#               precision    recall  f1-score   support
# 
#      ANORMAL       0.99      0.98      0.99     12447
#       NORMAL       0.97      0.98      0.98      6553
# 
#     accuracy                           0.98     19000
#    macro avg       0.98      0.98      0.98     19000
# weighted avg       0.98      0.98      0.98     19000

#%%
""" Naive Bayes Classifier """
print('Testando o Naive Bayes Classifier')
previsoes_naive_bayes = naive_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_naive_bayes)}')
# 0.9378947368421052

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_naive_bayes)}')
# [[12228   219]
#  [  961  5592]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_naive_bayes)}')
#               precision    recall  f1-score   support
#
#      ANORMAL       0.93      0.98      0.95     12447
#       NORMAL       0.96      0.85      0.90      6553
#
#     accuracy                           0.94     19000
#    macro avg       0.94      0.92      0.93     19000
# weighted avg       0.94      0.94      0.94     19000

#%%
""" K-Nearest Neighbors Classifier """
print('Testando o K Nearest Neighbors Classifier')
previsoes_knn_classifier = knn_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_knn_classifier)}')
# 0.9203684210526316

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_knn_classifier)}')
# [[11189  1258]
#  [  255  6298]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_knn_classifier)}')
#               precision    recall  f1-score   support
#
#      ANORMAL       0.98      0.90      0.94     12447
#       NORMAL       0.83      0.96      0.89      6553
#
#     accuracy                           0.92     19000
#    macro avg       0.91      0.93      0.91     19000
# weighted avg       0.93      0.92      0.92     19000

#%%
""" Logistic Regression Classifier """
print('Testando o Logistic Regression Classifier')
previsoes_logistic_classifier = logistic_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_logistic_classifier)}')
# 0.9388421052631579

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_logistic_classifier)}')
# [[11858   589]
#  [  573  5980]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_logistic_classifier)}')
#               precision    recall  f1-score   support
#
#      ANORMAL       0.95      0.95      0.95     12447
#       NORMAL       0.91      0.91      0.91      6553
#
#     accuracy                           0.94     19000
#    macro avg       0.93      0.93      0.93     19000
# weighted avg       0.94      0.94      0.94     19000

#%%
""" Suport Vector Machine """
print('Testando o Suport Vector Machine Classifier')
previsoes_suport_vector_machine = svc_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_suport_vector_machine)}')
# 0.9470526315789474

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_suport_vector_machine)}')
# [[12076   371]
#  [  635  5918]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_suport_vector_machine)}')
#               precision    recall  f1-score   support
#
#      ANORMAL       0.95      0.97      0.96     12447
#       NORMAL       0.94      0.90      0.92      6553
# 
#     accuracy                           0.95     19000
#    macro avg       0.95      0.94      0.94     19000
# weighted avg       0.95      0.95      0.95     19000

#%%
""" Neural Network Classifier """
print('Testando o Neural Network Classifier')
previsoes_neural_network = neural_network_classifier.predict(previsores_teste)

print(f'\tAcuracia: {accuracy_score(classes_teste, previsoes_neural_network)}')
# 0.9747368421052631

print(f'\tMatriz de confusão:\n{confusion_matrix(classes_teste, previsoes_neural_network)}')
# [[ 3383   202]
#  [  280 51215]]

print(f'\tRelatório:\n{classification_report(classes_teste, previsoes_neural_network)}')
#               precision    recall  f1-score   support
# 
#      ANORMAL       0.98      0.98      0.98     12447
#       NORMAL       0.96      0.96      0.96      6553
# 
#     accuracy                           0.97     19000
#    macro avg       0.97      0.97      0.97     19000
# weighted avg       0.97      0.97      0.97     19000

#%%
print('Salvando os resultados dos testes no arquivo data_classifiers_test.pkl')
with open('data_classifiers_test.pkl', mode='wb') as file:
    pickle.dump([previsores_teste, classes_teste, previsoes_tree_classifier,
                 previsoes_naive_bayes, previsoes_knn_classifier,
                 previsoes_logistic_classifier, previsoes_suport_vector_machine,
                 previsoes_neural_network], file)
