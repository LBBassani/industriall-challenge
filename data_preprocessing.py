# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:43:03 2023

@author: lorena
"""
#%%
import pandas as pd

#%%
base = pd.read_csv('data/target_iALL_PS.csv')

for i in range(52):
    base_name = f'TAG_iALL_PS_{i:02d}'
    new_base_column = pd.read_csv(f'data/{base_name}.csv')
    base = pd.merge(base, new_base_column, on="timestamp")
    
#%%
base.to_csv('data/data_base.csv', encoding='utf-8', index=False)

#%%
base = pd.read_csv('data/data_base.csv')
base_description = base.describe()

"""Column 'TAG_iAL_PS_15' does not contain any values, so it will be dropped"""
base = base.drop(columns=['TAG_iALL_PS_15'])
base_description = base.describe()

#%%
base_nan_count = base.isna().sum()

""" Fill NaN with interpolated values (value based on estimation based on 
    the values of neighboring data points) instead of dropping the rows """
base = base.interpolate()
new_base_nan_count = base.isna().sum()

#%%
""" Visualization """
import seaborn as sns
import matplotlib.pyplot as plt

#%%

for i in range(1):
    base_name = f'TAG_iALL_PS_{i:02d}'
    grafico = sns.lineplot(data=base, x='timestamp', y=base_name, hue='target_iALL_PS')
    plt.clf()

#%%
previsores = base.iloc[:, 2:]
classe = base.iloc[:, 1:2]

#%%
"""padronização dos dados"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#%%
"""Separação entre base de treino e teste"""
from sklearn.model_selection import train_test_split
previsores_treino, previsores_teste, classes_treino, classes_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#%%
"""Salvar os dados preprocessados"""
import pickle

#%%

with open('data_preprocessed.pkl', mode='wb') as file:
    pickle.dump([previsores_treino, previsores_teste, classes_treino, classes_teste], file)
    
#%%
""" Save preprocessing scaler """
with open('data_scaler.pkl', mode='wb') as file:
    pickle.dump(scaler, file)