# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:32:52 2023

@author: lorena
"""

#%%
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#%%
base = pd.read_csv('data/preprocessed_base.csv')

#%%
""" Visualizing ANORMAL occurrences """
# From 17000 to 18999
anormal_occurance_area = base[17000:19000]
anormal_occurance = anormal_occurance_area[anormal_occurance_area['target_iALL_PS'].eq('ANORMAL')]
normal_occurance_before = anormal_occurance_area[anormal_occurance_area.index < anormal_occurance.index[0]]
normal_occurance_after = anormal_occurance_area[anormal_occurance_area.index > anormal_occurance.index[-1]].head(normal_occurance_before.shape[0])

#%%
for i in range(52):
    column_name = f'TAG_iALL_PS_{i:02d}'
    if column_name in base:
        sns.lineplot(data=normal_occurance_before, y=column_name, x=normal_occurance_before.index, color='deepskyblue')
        sns.lineplot(data=anormal_occurance, y=column_name, x=anormal_occurance.index, color='coral')
        sns.lineplot(data=normal_occurance_after, y=column_name, x=normal_occurance_after.index, color='deepskyblue')
        plt.show()
        plt.clf()
        
#%%
""" Visualizing ANORMAL occurrences """
# From 24000 to 27999
anormal_occurance_area = base[24000:28000]
anormal_occurance = anormal_occurance_area[anormal_occurance_area['target_iALL_PS'].eq('ANORMAL')]
normal_occurance_before = anormal_occurance_area[anormal_occurance_area.index < anormal_occurance.index[0]]
normal_occurance_after = anormal_occurance_area[anormal_occurance_area.index > anormal_occurance.index[-1]].head(normal_occurance_before.shape[0])

#%%
for i in range(52):
    column_name = f'TAG_iALL_PS_{i:02d}'
    if column_name in base:
        sns.lineplot(data=normal_occurance_before, y=column_name, x=normal_occurance_before.index, color='deepskyblue')
        sns.lineplot(data=anormal_occurance, y=column_name, x=anormal_occurance.index, color='coral')
        sns.lineplot(data=normal_occurance_after, y=column_name, x=normal_occurance_after.index, color='deepskyblue')
        plt.show()
        plt.clf()
        
#%%
""" Visualizing ANORMAL occurrences """
# From 127000 to 133999
anormal_occurance_area = base[127000:140000]
anormal_occurance = anormal_occurance_area[anormal_occurance_area['target_iALL_PS'].eq('ANORMAL')]
normal_occurance_before = anormal_occurance_area[anormal_occurance_area.index < anormal_occurance.index[0]]
normal_occurance_after = anormal_occurance_area[anormal_occurance_area.index > anormal_occurance.index[-1]].head(normal_occurance_before.shape[0])

#%%
for i in range(52):
    column_name = f'TAG_iALL_PS_{i:02d}'
    if column_name in base:
        sns.lineplot(data=normal_occurance_before, y=column_name, x=normal_occurance_before.index, color='deepskyblue')
        sns.lineplot(data=anormal_occurance, y=column_name, x=anormal_occurance.index, color='coral')
        sns.lineplot(data=normal_occurance_after, y=column_name, x=normal_occurance_after.index, color='deepskyblue')
        plt.show()
        plt.clf()