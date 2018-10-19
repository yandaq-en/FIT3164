import pandas as pd
import os


def impute(df, by, method='mean'):
    if method == 'mean':
        return df.fillna(by.mean())
    elif method == 'median':
        return df.fillna(by.median())
    else:
        raise ValueError("Imputation method not allowed!\n - Please choose from ['mean','median']")


def load_files(dir_name):
    dfs = []
    for file in os.listdir(dir_name):
        dfs.append(pd.read_csv(dir_name + file))
    return dfs


def long2wide(dfs, col, value, index=None):
    for i in range(len(dfs)):
        dfs[i] = dfs[i].pivot(index=index, columns=col, values=value)
    return dfs


def feature_extract(dfs, static_col):
    mean = pd.DataFrame()
    maximum = pd.DataFrame()
    minimum = pd.DataFrame()

    for df in dfs:
        mean = mean.append(df.mean(), ignore_index=True)
        maximum = maximum.append(df.max(), ignore_index=True)
        minimum = minimum.append(df.min(), ignore_index=True)

    static = mean[static_col]
    mean = mean.drop(static_col, axis=1).add_suffix('_mean')
    maximum = maximum.drop(static_col, axis=1).add_suffix('_max')
    minimum = minimum.drop(static_col, axis=1).add_suffix('_min')
    return pd.concat([static, mean, maximum, minimum], axis=1, sort=False)


def non0var(df):
    categorical = df.select_dtypes(include='object')
    numerical = df.select_dtypes(exclude='object')
    numerical = numerical.iloc[:, list(numerical.var() != 0)]
    df = pd.concat([categorical, numerical], axis=1)
    return df


def synchronize(train, validation, test):
    test = test.drop(list(set(test).difference(set(train))), axis=1)
    test = test.drop(list(set(test).difference(set(validation))), axis=1)
    validation = validation.drop(list(set(validation).difference(set(train))), axis=1)
    validation = validation.drop(list(set(validation).difference(set(test))), axis=1)
    train = train.drop(list(set(train).difference(set(test))), axis=1)
    train = train.drop(list(set(train).difference(set(validation))), axis=1)
    return train, validation, test
