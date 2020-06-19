################################################################################
###########################ANNA - SINA - FABIEN#################################
##########################MACHINE LEARNING TASK2################################
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier

# Because each patient has 12 consecutive hours recorded
# but not the first 12 we have to fix this
better_time = list(range(12))

# Loading the data from csv
# Then overwrite the time and set the index
train_features = pd.read_csv('data/train_features.csv')
train_features['Time'] = better_time * (len(train_features.index) // 12)
train_features.set_index(['pid', 'Time'], inplace=True)

# Loading the labels from csv and set the index to pid
train_labels = pd.read_csv(
    'data/train_labels.csv',
    index_col=['pid'],
)

# Same for the test data
test_features = pd.read_csv('data/test_features.csv')
test_features['Time'] = better_time * (len(test_features.index) // 12)
test_features.set_index(['pid', 'Time'], inplace=True)

# WARNING: Uncomment this part to save the patient data after one aggregation
# as it takes a long time to compile the list
patient_data = train_features.groupby(level=0).agg(list)
patient_data_test = test_features.groupby(level=0).agg(list)

patient_data.to_pickle("./patient_data.pkl")
patient_data_test.to_pickle("./patient_data_test.pkl")
patient_data = pd.read_pickle("./patient_data.pkl")
patient_data_test = pd.read_pickle("./patient_data_test.pkl")
print(patient_data, patient_data_test)

feature_list = patient_data.columns.tolist()[3:]
print(feature_list)

# Calculate averages to insert for nan's where there is no data
averages = train_features.mean(axis=0)

# Impute NaN values with interpolation
for dataset in [patient_data, patient_data_test]:
    for index, row in dataset.iterrows():
        for label, content in row.items():
            content = pd.Series(content)
            if content.dropna().empty:
                content.fillna(averages[label])
            else:
                content.interpolate(limit_direction='both')

patient_data.to_pickle("./patient_data-interpolated.pkl")
patient_data_test.to_pickle("./patient_data_test-interpolated.pkl")
patient_data = pd.read_pickle("./patient_data-interpolated.pkl")
patient_data_test = pd.read_pickle("./patient_data_test-interpolated.pkl")
print(patient_data, patient_data_test)

# Extract features
def create_features(dataset):
    X = []
    for index, row in patient_data.iterrows():
        feature_row = []
        for label, content in row.items():
            feature_row.append(np.mean(content))
            feature_row.append(content[-1]-content[0])
            feature_row.append(np.min(content))
            feature_row.append(np.max(content))
            feature_row.append(np.std(content))
        X.append(feature_row)
    return np.asarray(X)

X = create_features(patient_data)
X_test = create_features(patient_data_test)
y_pred_all = []
y_pred_all.append(list(patient_data_test.keys()))
print(X, X_test)

# Predict test set
for label in train_labels.columns[1:11]:
    brfc = BalancedRandomForestClassifier(
        random_state=0,
        n_estimators=250,
        max_depth=90,
        max_features='sqrt',
        sampling_strategy=1,
        bootstrap=False
    ).fit(X, train_labels[label])
    y_pred=brfc.predict(X_test)
    y_pred_all.append(y_pred)

brfc = BalancedRandomForestClassifier(
    random_state=0,
    n_estimators=400,
    max_depth=100,
    max_features='sqrt',
    sampling_strategy=1,
    bootstrap=False
).fit(X, train_labels['LABEL_Sepsis'])

y_pred = brfc.predict(X_test)
y_pred_all.append(y_pred)


for label in train_labels.columns[12:16]:
    rfr = RandomForestRegressor(
        random_state=0,
        n_estimators=300,
        max_depth=150,
        max_features=0.4
    ).fit(X, train_labels[label])
    y_pred = rfr.predict(X_test)
    y_pred_all.append(y_pred)


df_result = pd.DataFrame(np.transpose(y_pred_all), None, train_labels.columns)
df_result.to_csv(
    'results1.zip',
    index=False,
    float_format='%.3f',
    compression='zip'
)