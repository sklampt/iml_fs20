################################################################################
###########################ANNA - SINA - FABIEN#################################
##########################MACHINE LEARNING TASK2################################
################################################################################

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load training data - labels
trainl = np.loadtxt(
    'data/train_labels.csv',
    delimiter=',',
    skiprows=1,
    dtype=np.double,
)
id_trainl = trainl[:, 0].astype(np.int)
x_trainl = trainl[:, 1:]

# Load training data - features
train = np.loadtxt(
    'data/train_features.csv',
    delimiter=',',
    skiprows=1,
    dtype=np.double,
)
id_train = train[:, 0].astype(np.int)
x_train = train[:, 1:]

# Load test data - features
test = np.loadtxt(
    'data/test_features.csv',
    delimiter=',',
    skiprows=1,
    dtype=np.double,
)
id_test = test[:, 0].astype(np.int)
x_test = test[:, 1:]

# Debug part...
# train the model with KNeighborsClassifier
DEBUG = True

# DEBUG = False
if (DEBUG):
    kf = KFold(n_splits=5)

    results = []
    parameters = np.arange(0.01,1,0.01)
    i = 0
    best_index = 0
    best_score = 0

    for parameter in parameters:
        acc_sum = 0

        for train, test in kf.split(x_train):
            classifier_model = NuSVC(
                nu=parameter,
                degree=3,
                gamma='scale',
                tol=0.001,
                coef0=0.0,
                decision_function_shape='ovr',
                kernel='rbf'
            ).fit(x_train[train],y_train[train])

            y_pred = classifier_model.predict(x_train[test])
            acc_iterating = accuracy_score(y_train[test], y_pred)
            acc_sum += acc_iterating

        results.append(acc_sum / 5)

        if(best_score < acc_sum / 5):
            best_score = acc_sum / 5
            best_index = i

        print(
            '[' + '{:5.1f}'.format(i / len(parameters) * 100) + '%]',
            end="\r"
        )
        i += 1

    fig = plt.figure()
    plt.title('NuSVC: Kernel "rbf"')
    ax = fig.add_subplot(111)
    ax.plot(parameters, results)
    ax.axvline(x=parameters[best_index], color="r", label=parameters[best_index])
    ax.legend()
    plt.savefig('result.png')

    print(best_score, parameters[best_index])

# Train the model
classifier_model = NuSVC(
    nu=0.309,
    degree=3,
    gamma='scale',
    tol=0.001,
    coef0=0.0,
    decision_function_shape='ovr',
    kernel='rbf'
).fit(x_train,y_train)

# Predict test data with the trained model
y_pred = classifier_model.predict(x_test)

# Save the prediction in the result file
result = np.array([id_test, y_pred]).transpose()

np.savetxt(
    'result.csv',
    result,
    header="Id,y",
    comments='',
    delimiter=",",
    fmt="%i,%s",
)