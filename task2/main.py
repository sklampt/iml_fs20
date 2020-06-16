################################################################################
###########################ANNA - SINA - FABIEN#################################
##########################MACHINE LEARNING TASK2################################
################################################################################

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Because each patient has 12 consecutive hours recorded
# but not the first 12 we have to fix this
better_time = list(range(12))

# Loading the data from csv
# Then overwrite the time and set the index
train_features = pd.read_csv(
    'data/train_features.csv',
)
train_features['Time'] = better_time * (len(train_features.index) // 12)
train_features.set_index(['pid', 'Time'], inplace=True)

# Loading the labels from csv and set the index to pid
train_labels = pd.read_csv(
    'data/train_labels.csv',
    index_col=['pid'],
)

# Same for the test data
test_features = pd.read_csv(
    'data/test_features.csv',
)
test_features['Time'] = better_time * (len(test_features.index) // 12)
test_features.set_index(['pid', 'Time'], inplace=True)


# For subtask 1 we have to predict some labels, we assume that they are
# only relying on their feature counterpart, ex. BaseExcess -> LABEL_BaseExcess
subtask1 = [
    'BaseExcess',
    'Fibrinogen',
    'AST',
    'Alkalinephos',
    'Bilirubin_total',
    'Lactate',
    'TroponinI',
    'SaO2',
    'Bilirubin_direct',
    'EtCO2',
]
for feature in subtask1:
    X = train_features[[feature]].stack(dropna=False).unstack(level=1)
    X_hat = test_features[[feature]].stack(dropna=False).unstack(level=1)
    y = train_labels['LABEL_'+feature]

    imp = SimpleImputer(strategy='mean', copy=False)
    imp.fit_transform(X)
    imp.fit_transform(X_hat)

    # print(X, X_hat, y)
    # exit(1)

    clf = SVC(kernel='sigmoid', class_weight='balanced')
    clf.fit(X, y)

    y_hat = clf.predict(X_hat)

    # TODO: Save the predictions in the correct way...
    print(y_hat)


# For subtask 2 we have to predict sepsis
# Symptoms for sepsis include fever, trouble breathing, low blood pressure
# and high heart rate. Therefore we use each of these as features, not
# as time series but as mean, variance, min, max and gradient.


# Subtask 3: Predict some vitals:
# 