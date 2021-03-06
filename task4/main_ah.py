################################################################################
############################SINA - FABIEN - ANNA################################
###########################MACHINE LEARNING TASK4###############################
################################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
#from PIL import Image
from tensorflow import keras
from tensorflow.keras import preprocessing

from sklearn.svm import SVC


# Loading train data from csv
train_triplets = pd.read_csv(
    'data/train_triplets.txt',
    header=None,
)

# Same for test data
test_triplets = pd.read_csv(
    'data/test_triplets.txt',
    header=None,
)

# Initialize train and test matrix 
X = [[0 for x in range(3)] for y in range(59515)]
X_hat = [[0 for x in range(3)] for y in range(59544)]

# Create train set with jpg files
rcounter = 0
for index, line in train_triplets.iterrows():
    counter = 0
    for i in line[0].split(' '):
        image = tf.keras.preprocessing.image.load_img('data/food/'+i+'.jpg')
        input_arr = keras.preprocessing.image.img_to_array(image)
        X[rcounter][counter] = np.array([input_arr])
        counter += 1
    rcounter += 1
print(X)
exit()

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# First Try: Impute and SVC with sigmoid function
# DOES NOT WORK (maybe its the imputer)
# TODO: Try with numpy...
# imp = SimpleImputer(strategy='constant', fill_value=0, copy=False)
# imp.fit_transform(X_train)
# imp.fit_transform(X_test)

# # print(X, X_hat, y)
# # exit(1)

# clf = SVC(kernel='sigmoid', class_weight='balanced')
# clf.fit(X_train, y_train)

# y_hat = clf.predict(X_test)

# Second Try: Tests will be only if there has one yet...
# DOES NOT WORK
y_hat = pd.DataFrame(X_test.isnull().sum(axis=1) < 12).astype(int).to_numpy()

# TODO: Next Idea would be to use the vitals... could also be useful for sepsis

# TODO: Save the predictions in the correct way...
print(roc_auc_score(y_test, y_hat))


# For subtask 2 we have to predict sepsis
# Symptoms for sepsis include fever, trouble breathing, low blood pressure
# and high heart rate. Therefore we use each of these as features, not
# as time series but as mean, variance, min, max and gradient.


# Subtask 3: Predict some vitals:
# 