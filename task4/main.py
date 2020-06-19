################################################################################
############################SINA - FABIEN - ANNA################################
###########################MACHINE LEARNING TASK4###############################
################################################################################

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras

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

# crop all img to same size, manuell mit PIL, 200x200 np.array, train_on_batch
# generator 

# Initialize train and test matrix 
X = [[0 for x in range(3)] for y in range(59515)]
X_hat = [[0 for x in range(3)] for y in range(59544)]

# Create train set with jpg files
rcounter = 0
for index, line in train_triplets.iterrows():
    counter = 0
    for i in line[0].split(' '):
        # print(i)
        X[rcounter][counter] = np.array(Image.open('data/food/'+i+'.jpg').getdata())
        counter += 1
    rcounter += 1
print(X)
exit()

# Same for test set
rcounter = 0
for index, line in test_tripletsiterrows():
    counter = 0
    for i in line[0].split(' '):
        #print(i)
        X[rcounter][counter] = np.array(Image.open('data/food/'+i+'.jpg').getdata())
        counter += 1
    rcounter += 1
            
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)