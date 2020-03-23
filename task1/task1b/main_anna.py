import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn import linear_model

#load from train
train = np.loadtxt (
    'train.csv',
    delimiter=',',
    skiprows=1,
)

y_train = train[:, 1]
X_lin = train[:, 2:]
X_quad = np.square(X_lin)
X_exp = np.exp(X_lin)
X_cos = np.cos(X_lin)
X_con = np.ones([X_lin.shape[0], 1])

X_train = np.block([X_lin, X_quad, X_exp, X_cos, X_con])

model = linear_model.LassoCV(fit_intercept=False, max_iter=20000)
model.fit(X_train, y_train)

for i in model.coef_:
    print (i)
