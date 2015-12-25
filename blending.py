
"""
inspired by the code of Emanuele Olivetti

needs some fixes
"""

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import re

from sklearn.metrics import classification_report

#%% Settings
attribute = "_volume"
np.random.seed(0) # seed to shuffle the train set
n_folds = 10
verbose = True
shuffle = False

#%% Data
data = pd.read_csv(os.path.join('..', '..', 'data', 'FS_imputed.csv'), index_col=0)
data = data[data['sex']==1]
y = data['control']

if (attribute == "_volume" or attribute == "ALL"):
        volume_columns = filter(lambda x:"_volume" in x, data.iloc[:,10:].columns)
        eTIV = data['EstimatedTotalIntraCranialVol_volume']
        data[volume_columns] = data[volume_columns].div(eTIV, axis='index')
        data.drop('EstimatedTotalIntraCranialVol_volume', axis=1, inplace=True)
if attribute == "ALL":
    ALL = "ALL"
    X = data
else:
    ALL = ""
    attribute_columns = filter(lambda x:re.search(attribute,x), data.columns)
    X = data[attribute_columns]



folds = StratifiedKFold(y, n_folds = n_folds, shuffle = True, random_state = np.random.seed(0))

for i, (train_index, test_index) in enumerate(folds):
    if i <7: 
        continue
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clfs = [RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='entropy')]
            # GradientBoostingClassifier(n_estimators=200,learning_rate=0.01, subsample=0.5, max_depth=6)]


    print "Creating train and test sets for blending."  # caution: print is a function in python 3
    skf = list(StratifiedKFold(y_train,  n_folds = n_folds, shuffle = False, random_state = np.random.seed(0)))

    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
        for k, (train_index_j, test_index_j) in enumerate(skf):
            print "Fold", k
            X_train_k, X_test_k = X_train.iloc[train_index_j], X_train.iloc[test_index_j]
            y_train_k, y_test_k = y_train[train_index_j], y_train[test_index_j]
            clf.fit(X_train_k, y_train_k)
            dataset_blend_train[test_index_j, j] = clf.predict_proba(X_test_k)[:,1]
            dataset_blend_test_j[:, k] = clf.predict_proba(X_test)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    print dataset_blend_train.shape

    clf.fit(dataset_blend_train, y_train)
    y_predicted = clf.predict(dataset_blend_test)
    print y_predicted
    print classification_report(y_test,y_predicted)
    print clf.score(dataset_blend_test, y_test)

