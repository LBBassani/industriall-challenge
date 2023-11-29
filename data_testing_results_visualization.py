# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:00:59 2023

@author: lorena
"""

#%%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#%%
with open('data_classifiers_test.pkl', mode='rb') as file:
    previsores_teste, classes_teste, previsoes_tree_classifier, previsoes_naive_bayes, previsoes_knn_classifier, previsoes_logistic_classifier, previsoes_suport_vector_machine, previsoes_neural_network = pickle.load(file)

#%%
previsores_teste_dataframe = pd.DataFrame(previsores_teste[:1300])
previsores_teste_dataframe['real_class'] = classes_teste.ravel()[:1300]
previsores_teste_dataframe['tree_classified'] = previsoes_tree_classifier[:1300]
previsores_teste_dataframe['knn_classified'] = previsoes_knn_classifier[:1300]
previsores_teste_dataframe['neural_classified'] = previsoes_neural_network[:1300]
previsores_teste_dataframe['group'] = previsores_teste_dataframe['real_class'].ne(previsores_teste_dataframe['real_class'].shift()).cumsum()
previsores_teste_dataframe= previsores_teste_dataframe.groupby('group')
data_set = []
for name, data in previsores_teste_dataframe:
    data_set.append(data)

#%%
""" Best classifier errors """
for i in range(np.shape(previsores_teste)[1]):
    fig, ax = plt.subplots()
    ax.set_title('Tree Classifier Errors')
    ax.set_ylim([-5, 5])
    ax.set_xlim([-100, 1400])
    total_errors = 0
    for data in data_set:
        middle = data.index[data.shape[0]//2] - 150
        p1 = sns.lineplot(ax=ax, zorder=1, data=data, y=i, x=data.index, color=('coral' if 'ANORMAL' in data['real_class'].unique() else 'deepskyblue'))
        error_class = data[data['real_class'].ne(data['tree_classified'])]
        errors_number = error_class.size
        plt.text(middle, 3, f'{errors_number} errors')
        total_errors = total_errors + errors_number
        p2= sns.scatterplot(ax=ax, zorder=2, data=error_class, y=i, x=error_class.index, color=('darkblue' if 'ANORMAL' in error_class['real_class'].unique() else 'firebrick'))
   
    plt.text(-50, 4, f'Total errors: {total_errors}')
    plt.show()
    plt.clf()

#%%
""" Worst classfier errors """
for i in range(np.shape(previsores_teste)[1]):
    fig, ax = plt.subplots()
    ax.set_title('KNN Classifier Errors')
    ax.set_ylim([-5, 5])
    ax.set_xlim([-100, 1400])
    total_errors = 0
    for data in data_set:
        middle = data.index[data.shape[0]//2] - 150
        p1 = sns.lineplot(ax=ax, zorder=1, data=data, y=i, x=data.index, color=('coral' if 'ANORMAL' in data['real_class'].unique() else 'deepskyblue'))
        error_class = data[data['real_class'].ne(data['knn_classified'])]
        errors_number = error_class.size
        plt.text(middle, 3, f'{errors_number} errors')
        total_errors = total_errors + errors_number
        p2= sns.scatterplot(ax=ax, zorder=2, data=error_class, y=i, x=error_class.index, color=('darkblue' if 'ANORMAL' in error_class['real_class'].unique() else 'firebrick'))
    plt.show()
    plt.clf()

#%%
""" Second best classfier errors """
for i in range(np.shape(previsores_teste)[1]):
    fig, ax = plt.subplots()
    ax.set_title('Neural Network Classifier Errors')
    ax.set_ylim([-5, 5])
    ax.set_xlim([-100, 1400])
    total_errors = 0
    for data in data_set:
        middle = data.index[data.shape[0]//2] - 150
        p1 = sns.lineplot(ax=ax, zorder=1, data=data, y=i, x=data.index, color=('coral' if 'ANORMAL' in data['real_class'].unique() else 'deepskyblue'))
        error_class = data[data['real_class'].ne(data['neural_classified'])]
        errors_number = error_class.size
        plt.text(middle, 3, f'{errors_number} errors')
        total_errors = total_errors + errors_number
        p2= sns.scatterplot(ax=ax, zorder=2, data=error_class, y=i, x=error_class.index, color=('darkblue' if 'ANORMAL' in error_class['real_class'].unique() else 'firebrick'))
    plt.show()
    plt.clf()
