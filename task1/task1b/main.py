################################################################################
############################SINA - FABIEN - ANNA################################
##########################MACHINE LEARNING TASK1A###############################
################################################################################

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# Load training data
#
# np.set_printoptions(precision=16)#set precision for proper data loading
train = np.loadtxt(
    'data/train.csv',
    delimiter=',',
    skiprows=1,
    dtype=np.float, # change to longdouble doesn't effect anything
)

y_train = train[:, 1]
x_train = train[:, 2:]

x_train_sq = np.square(x_train)
x_train_exp = np.exp(x_train)
x_train_cos = np.cos(x_train)
ones = np.ones([x_train.shape[0], 1])

x_train_extended=np.block([x_train,x_train_sq,x_train_exp,x_train_cos,ones])


# Train the model
#
model = Ridge(alpha=1000).fit(x_train_extended, y_train)
weights_hat = model.coef_
weights_hat[20] = model.intercept_

for i in range(21):
    print(weights_hat[i])

# Calculate the Root Mean Squared Error with the known function
# This only works, because we know the correct prediction function
#
# weights_sample =np.arange(1,22)
#
# RMSE = mean_squared_error(weights_sample, weights)**0.5
# RMSE_hat = mean_squared_error(weights_sample, weights_hat)**0.5
# print('---')
# print(RMSE, RMSE_hat)
