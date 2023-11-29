# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:43:03 2023

@author: lorena
"""
#%%
import pandas as pd

import pickle

#%%
print('Lendo dados dos sensores...')
base = pd.read_csv('data/target_iALL_PS.csv')

for i in range(52):
    base_name = f'TAG_iALL_PS_{i:02d}'
    print(f'\tLendo arquivo data/{base_name}.csv')
    new_base_column = pd.read_csv(f'data/{base_name}.csv')
    base = pd.merge(base, new_base_column, on="timestamp")
    print(f'\tLeitura do arquivo data/{base_name}.csv finalizada')
    
#%%
print('Salvando dados no arquivo data/data_base.csv')
base.to_csv('data/data_base.csv', encoding='utf-8', index=False)

#%%
""" Uncomment if the file data/data_base.csv alredy exists """
# print('Lendo dados do arquivo data/data_base.csv') 
# base = pd.read_csv('data/data_base.csv')
print('Descrição da base de dados')
base_description = base.describe()
print(base_description)

"""Column 'TAG_iAL_PS_15' does not contain any values, so it will be dropped"""
print('Drop da coluna TAG_iALL_PS_15 da base de dados por não conter valores')
base = base.drop(columns=['TAG_iALL_PS_15'])
base_description = base.describe()

#%%
base_nan_count = base.isna().sum()
print(f'Quantidade de atributos nulos na base:\n{base_nan_count}')

""" Fill NaN with interpolated values (value based on estimation based on 
    the values of neighboring data points) instead of dropping the rows """

print('Interpolando valores nulos')
base = base.interpolate()

new_base_nan_count = base.isna().sum()
print(f'Quantidade de atributos nulos na base:\n{new_base_nan_count}')

#%%
print('Salvando dados no arquivo data/preprocessed_base.csv')
base.to_csv('data/preprocessed_base.csv', encoding='utf-8', index=False)

#%%
print('Separando base entre previsores (atributos de entrada) e classes (atributos de saída)')
previsores = base.iloc[:, 2:]
classe = base.iloc[:, 1:2].values.ravel()

#%%
""" data scaling """
from sklearn.preprocessing import StandardScaler
print('Normalizando os dados previsores')
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#%%
""" saving preprocessed data"""
print('Salvando os dados normalizados no arquivo data/data_normalized.pkl')
with open('data/data_normalized.pkl', mode='wb') as file:
    pickle.dump([previsores, classe], file)

#%%
"""Separação entre base de treino e teste"""
from sklearn.model_selection import train_test_split
print('Separando dados entre base de treino e teste')
previsores_treino, previsores_teste, classes_treino, classes_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#%%
print('Salvando os dados separados para teste e treino no arquivo data_preprocessed.pkl')
with open('data_preprocessed.pkl', mode='wb') as file:
    pickle.dump([previsores_treino, previsores_teste, classes_treino, classes_teste], file)
    
#%%
""" Save preprocessing scaler """
print('Salvando o normalizador para aplicar nas entradas dos modelos no arquivo data_scaler.pkl')
with open('data_scaler.pkl', mode='wb') as file:
    pickle.dump(scaler, file)