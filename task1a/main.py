################################################################################
################################SINA - FABIEN###################################
##########################MACHINE LEARNING TASK1A###############################
################################################################################


# import scripts
#
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# Load training data
#
train = np.loadtxt(
    'data/train.csv',
    delimiter=',',
    skiprows=1,
    dtype=np.float #as clearly told in the handout
)
y_train = train[:, 1]
x_train = train[:, 2:]
kf = KFold(n_splits=10)
for l in (0.1,1,10,100,1000):
    sum = 0
    # train model
    #
    for train_index, test_index in kf.split(x_train):
        model = Ridge(alpha=l).fit(x_train[train_index],y_train[train_index])

        # predict the model
        #
        y_pred=model.predict(x_train[test_index])
        # sum the RMSE
        RMSE = mean_squared_error(y_train[test_index], y_pred)**0.5
        sum += RMSE
    # print the solution
    #
    print(sum/kf.get_n_splits(x_train))


#export result with | tee result.csv
