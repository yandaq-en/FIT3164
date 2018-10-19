# import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from hyperopt import hp, fmin, tpe
import os
import warnings
import random

random.seed(499)
import shap


########################################################################################################################
########################################################################################################################
##################################### functions which are used for preprocessing #######################################


# function which imputes missing values by mean or median values
def impute(df, by, method='mean'):
    if method == 'mean':
        return df.fillna(by.mean())
    elif method == 'median':
        return df.fillna(by.median())
    # if method not valid then raise error
    else:
        raise ValueError("Imputation method not allowed!\n - Please choose from ['mean','median']")


# load datasets into python
def load_files(dir_name):
    dfs = []
    for file in os.listdir(dir_name):
        dfs.append(pd.read_csv(dir_name + file))
    return dfs


# convert long format into wide format
def long2wide(dfs, col, value, index=None):
    for i in range(len(dfs)):
        dfs[i] = dfs[i].pivot(index=index, columns=col, values=value)
    return dfs


# function which extracts features from datasets and concat all datasets together
def feature_extract(dfs, static_col):
    mean = pd.DataFrame()
    maximum = pd.DataFrame()
    minimum = pd.DataFrame()
    # extract maximum, minimum and mean values
    for df in dfs:
        mean = mean.append(df.mean(), ignore_index=True)
        maximum = maximum.append(df.max(), ignore_index=True)
        minimum = minimum.append(df.min(), ignore_index=True)
    # leave static data out
    static = mean[static_col]
    mean = mean.drop(static_col, axis=1).add_suffix('_mean')
    maximum = maximum.drop(static_col, axis=1).add_suffix('_max')
    minimum = minimum.drop(static_col, axis=1).add_suffix('_min')
    # concat three dataframes forming the entire data used for train, test or validation
    return pd.concat([static, mean, maximum, minimum], axis=1, sort=False)

# remove all features with 0 variance
def non0var(df):
    # first need to separate categorical data and numerical data
    categorical = df.select_dtypes(include='object')
    numerical = df.select_dtypes(exclude='object')
    # only numerical data have variance
    numerical = numerical.iloc[:, list(numerical.var() != 0)]
    df = pd.concat([categorical, numerical], axis=1)
    return df


# function which makes three dataframes the same
def synchronize(train, validation, test):
    # delete columns which are not common
    test = test.drop(list(set(test).difference(set(train))), axis=1)
    test = test.drop(list(set(test).difference(set(validation))), axis=1)
    validation = validation.drop(list(set(validation).difference(set(train))), axis=1)
    validation = validation.drop(list(set(validation).difference(set(test))), axis=1)
    train = train.drop(list(set(train).difference(set(test))), axis=1)
    train = train.drop(list(set(train).difference(set(validation))), axis=1)
    return train, validation, test


########################################################################################################################
########################################################################################################################
###################################### functions which are used for modelling ##########################################


# function which is used to tune hyperparameters, can be a customized function
def objective(params):
    bst = xgb.XGBClassifier(max_depth=int(params['max_depth']), learning_rate=params['learning_rate'],
                            n_estimators=int(params['n_estimators']), gamma=params['gamma'],
                            min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'],
                            subsample=params['subsample'],
                            reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],
                            scale_pos_weight=params['scale_pos_weight'])
    bst.fit(train_X, train_y)
    yhat = bst.predict(validation_X)
    # use the same evaluation metric with the competition
    tn, fp, fn, tp = metrics.confusion_matrix(validation_y, yhat).ravel()
    # optimization can only find minimum hence use 1 - score
    return 1 - min(tp / (tp + fn), tp / (tp + fp))


# evaluate the model by testing data
def model_evaluation(model):
    model.fit(train_X, train_y)
    yhat = model.predict(test_X)
    # use the same metrics with the comprtition
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, yhat).ravel()
    return min(tp / (tp + fn), tp / (tp + fp)), model


########################################################################################################################
########################################################################################################################
######################################### high-level functions for the project #########################################


# the default preprocessing strategy
def default_preprocess(dir_name):
    # load data
    dfs = load_files(dir_name)
    # change format from long to wide
    dfs = long2wide(dfs, 'Parameter', 'Value')
    # extract features
    df = feature_extract(dfs, ['RecordID', 'Gender', 'Age', 'Height', 'ICUType'])
    # replace -1 by missing
    df = df.replace(-1, np.NaN)
    return df


# the default strategy for modelling
def default_modelling(space, objective, max_evals):
    warnings.filterwarnings("ignore")
    # find the best hyperparameters
    param = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, rstate=np.random.RandomState(499))
    # write to a text file for the use of application
    df = open('param.txt', 'w')
    df.write(str(param))
    df.close()
    # construct the model according to the hyperparameters chosen
    model = xgb.XGBClassifier(max_depth=int(param['max_depth']), learning_rate=param['learning_rate'],
                              n_estimators=int(param['n_estimators']), gamma=param['gamma'],
                              min_child_weight=param['min_child_weight'], max_delta_step=param['max_delta_step'],
                              subsample=param['subsample'], reg_alpha=param['reg_alpha'],
                              reg_lambda=param['reg_lambda'],
                              scale_pos_weight=param['scale_pos_weight'])
    # evaluate the model and print the score1
    score1, model = model_evaluation(model)
    return score1, model


########################################################################################################################
########################################################################################################################
##################################### functions for explainer ##########################################################


# construct the explainer using the model and testing data
def construct_explainer(model, test_X):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_X)
    return explainer, shap_values


# plot the summary for top features
def summary(shap_values, test_X):
    shap.summary_plot(shap_values, test_X)


# plot the effect for specified feature
def effect(feature_name, shap_values, test_X):
    shap.dependence_plot(feature_name, shap_values, test_X, interaction_index=None)


########################################################################################################################
########################################################################################################################
######################################## main function of the project ##################################################

if __name__ == '__main__':
    # load datasets
    train_X = default_preprocess('set-a/')
    train_X = impute(non0var(train_X), train_X)
    print('Training set ready to use!')
    validation_X = default_preprocess('set-b/')
    validation_X = impute(non0var(validation_X), validation_X)
    print('Validation set ready to use!')
    test_X = default_preprocess('set-c/')
    test_X = impute(non0var(test_X), train_X)
    print('test set ready to use!')
    train_X, validation_X, test_X = synchronize(train_X, validation_X, test_X)
    print('Datasets sychronized!')

    # load labels
    train_y = pd.read_csv('Outcomes-a.txt')[['RecordID', 'In-hospital_death']]
    validation_y = pd.read_csv('Outcomes-b.txt')[['RecordID', 'In-hospital_death']]
    test_y = pd.read_csv('Outcomes-c.txt')[['RecordID', 'In-hospital_death']]

    # merge datasets
    train = pd.merge(train_X, train_y, on='RecordID')
    validation = pd.merge(validation_X, validation_y, on='RecordID')
    test = pd.merge(test_X, test_y, on='RecordID')

    # generate X and y
    test_X = test.drop(['In-hospital_death', 'RecordID'], axis=1)
    test_X = test_X.reindex(sorted(test_X), axis=1)
    train_X = train.drop(['In-hospital_death', 'RecordID'], axis=1)
    train_X = train_X.reindex(sorted(train_X), axis=1)
    validation_X = validation.drop(['In-hospital_death', 'RecordID'], axis=1)
    validation_X = validation_X.reindex(sorted(validation_X), axis=1)
    test_y = test['In-hospital_death']
    train_y = train['In-hospital_death']
    validation_y = validation['In-hospital_death']

    # define the search space
    space = {'max_depth': hp.quniform('max_depth', 2, 5, 1),
             'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
             'n_estimators': hp.quniform('n_estimators', 80, 150, 1),
             'gamma': hp.uniform('gamma', 0, 10),
             'min_child_weight': hp.uniform('min_child_weight', 0, 5),
             'max_delta_step': hp.uniform('max_delta_step', 0, 10),
             'subsample': hp.uniform('subsample', 0.5, 1),
             'reg_alpha': hp.uniform('reg_alpha', 0, 10),
             'reg_lambda': hp.uniform('reg_lambda', 0, 10),
             'scale_pos_weight': hp.uniform('scale_pos_weight', 3, 5)
             }
    print('Model starts tuning!')

    # tune the model such that it reach its best performance
    score1, model = default_modelling(space, objective, 200)
    # show score1
    print(score1)

    # construct explainer
    explainer, shap_values = construct_explainer(model, test_X)
    # plot the summary for top features
    summary(shap_values, test_X)
    # plot the effect of age
    effect('Age', shap_values, test_X)
    # to see the individual explanation, one must use Juypter notebook
    shap.force_plot(explainer.expected_value, shap_values[1000,:], test_X.iloc[1000,:])