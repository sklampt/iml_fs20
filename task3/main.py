################################################################################
############################SINA - FABIEN - ANNA################################
###########################MACHINE LEARNING TASK3###############################
################################################################################

import string
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv(
    'data/train.csv',
)

test = pd.read_csv(
    'data/test.csv',
)

X = train['Sequence'].apply(lambda x: pd.Series(list(x)))
X_hat = test['Sequence'].apply(lambda x: pd.Series(list(x)))
print(X)

enc = OneHotEncoder()
enc.fit(X)
onehotlabels = enc.transform(X).toarray()
train_hotlabels = enc.transform(X_hat).toarray()


clf = MLPClassifier(random_state=1, max_iter=300).fit(onehotlabels, train['Active'])
clf.fit(onehotlabels, train['Active'])
y_pred = clf.predict(train_hotlabels)

print(y_pred)