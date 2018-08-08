import pandas as pd
import numpy as np
import os

dfs = []
for filename in os.listdir('..\\data\\trainset_A'):
    filename = '..\\data\\trainset_A\\' + filename
    dfs.append(pd.read_csv(filename, sep=','))

dfs1 = []
dfs2 = []
for i in range(len(dfs)):
    time_i = dfs[i]['Time']
    dfs[i] = pd.concat([time_i, dfs[i].pivot(columns='Parameter', values='Value')], axis=1).replace(-1, np.NaN).groupby(
        'Time').agg(np.mean)
    dfs[i] = dfs[i].fillna(dfs[i].mean())
    if dfs[i]['Gender'][0] == 0:
        dfs1.append(dfs[i])
    else:
        dfs2.append(dfs[i])

df1 = pd.concat(dfs1, sort=False)
df2 = pd.concat(dfs2, sort=False)
df1 = df1.fillna(df1.mean())
df2 = df2.fillna(df2.mean())
df = pd.concat([df1, df2], sort=False)

df.to_csv('mean_impute.csv')
outcome = pd.read_csv('..\\data\\outcome_A.txt')
data = pd.read_csv('mean_impute.csv')
df = pd.merge(data, outcome, on='RecordID', sort=False)
df.to_csv('merged_mean_impute.csv')
