################################################################################
############################SINA - FABIEN - ANNA################################
###########################MACHINE LEARNING TASK3###############################
################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

# Data transformation
def transform_data(dataset):
    num_rows = len(dataset.Sequence)

    x1 = np.array(dataset.Sequence[0][0])
    x2 = np.array(dataset.Sequence[0][1])
    x3 = np.array(dataset.Sequence[0][2])
    x4 = np.array(dataset.Sequence[0][3])

    for i in range(1,num_rows):
        x1 = np.append(x1,dataset.Sequence[i][0])
        x2 = np.append(x2,dataset.Sequence[i][1])
        x3 = np.append(x3,dataset.Sequence[i][2])
        x4 = np.append(x4,dataset.Sequence[i][3])

    numpy_data = np.array([x1,x2,x3,x4])

    df = pd.DataFrame(data=numpy_data, index=["x1","x2","x3","x4"])
    return df.transpose()

# Load as proposed
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train.Active.to_numpy()
X = transform_data(train)
X_hat = transform_data(test)

# Create transformer and classifier
# Combine it in pipeline
transformer = make_column_transformer((OneHotEncoder(),['x1','x2','x3','x4']))
classifier = MLPClassifier()

pipeline = make_pipeline(
    transformer,
    classifier,
)

# Finally, train model and predict test set
pipeline.fit(X,y)
y_hat = pipeline.predict(X_hat)

# Output results into file
pd.DataFrame(y_hat).to_csv(
    "results.csv",
    index=False,
    header=False,
)