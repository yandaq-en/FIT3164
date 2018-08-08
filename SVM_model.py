import pandas as pd
import numpy as np
from sklearn.svm import SVC

df = pd.read_csv('merged_mean_impute.csv')
testdf = pd.read_csv('test.csv')

dfcopy = df.groupby('RecordID').agg(np.mean)
dftest = testdf.groupby('RecordID').agg(np.mean)

dfcopy = dfcopy.reset_index()
train_X = dfcopy.drop(['Unnamed: 0','SAPS-I','SOFA','Length_of_stay','Survival','In-hospital_death','RecordID','ICUType'],axis = 1)
train_y = dfcopy['In-hospital_death']

dftest = dftest.reset_index()
test_X = dftest.drop(['Unnamed: 0','SAPS-I','SOFA','Length_of_stay','Survival','In-hospital_death','RecordID','ICUType'],axis = 1)
test_y = dftest['In-hospital_death']

clf = SVC(kernel='poly')
clf.fit(train_X, train_y)
print('Fit finished')
y_pre = clf.predict(test_X)
print('predict finished')
TPpFN = sum(y_pre)
TPpFP = sum(test_y)
TP = 0
for i in range(len(y_pre)):
    if list(y_pre)[i] == 1:
        if list(test_y)[i] == 1:
            TP += 1
SE = TP/TPpFN
PP = TP/TPpFP
print(min(SE,PP))