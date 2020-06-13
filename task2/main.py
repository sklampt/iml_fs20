################################################################################
###########################ANNA - SINA - FABIEN#################################
##########################MACHINE LEARNING TASK2################################
################################################################################

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

train_features = pd.read_csv(
    'data/train_features.csv',
    # index_col=['pid', 'Time'],
)

train_labels = pd.read_csv(
    'data/train_labels.csv',
    index_col=['pid'],
)

test_features = pd.read_csv(
    'data/test_features.csv',
    index_col=['pid', 'Time'],
)

train_features['better_time'] = list(range(12)) * 18995
train_features.set_index(['pid', 'better_time'], inplace=True)

X = train_features[['BaseExcess']].stack(dropna=False).unstack(level=1)
y = train_labels[['LABEL_BaseExcess']]

# X = X.transpose()
imp = SimpleImputer(strategy='mean', copy=False)
imp.fit_transform(X)
# X = X.transpose()

# print(X, y)
# exit(1)

clf = SVC(kernel='sigmoid', class_weight='balanced')

clf.fit(X, y)

# print(clf.predict(test_features))