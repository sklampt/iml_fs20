# To execute and save in result.csv file
# python3 main.py | tee result.csv

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

#load from train
train = np.loadtxt (
    'train.csv',
    delimiter=',',
    skiprows=1,
)

Y_train = train[:, 1]
X_train = train[:, 2:]

raw_model = Ridge()
parameters_for_raw_model = {'alpha':[0.01, 0.1, 1, 10, 100]}


gridsearch_model = GridSearchCV(
    raw_model,
    parameters_for_raw_model,
    scoring='neg_root_mean_squared_error',
    cv=10,
)

gridsearch_model.fit(X_train, Y_train)
# This function does approx. this:
# for fold in kfold(cv):
#     parameter --> raw model
#     model.fit
#     model.pred with test set
#     calculate score
# return the model with the best parameter


for score in gridsearch_model.cv_results_['mean_test_score']:
    print(-score)