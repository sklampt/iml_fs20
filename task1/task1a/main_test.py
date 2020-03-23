# To execute and save in result.csv file
# python3 main.py | tee result.csv

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

#load from train
train = np.loadtxt (
    'train.csv',
    delimiter=',',
    skiprows=1,
)

y = train[:, 1]
X = train[:, 2:]

alphas = np.array([0.01, 0.1, 1, 10, 100])

for x in alphas:
    model = Ridge(alpha = x)
    kf = KFold(n_splits=10, shuffle=False)
    RMSE_sum = 0.
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RMSE_sum += mean_squared_error(y_test, y_pred, squared=False)
    print(RMSE_sum/10)
