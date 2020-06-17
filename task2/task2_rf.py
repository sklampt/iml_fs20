import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import SVC,SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV

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
# patient_data = train_features.groupby(level=0).agg(list)
# patient_data_test = test_features.groupby(level=0).agg(list)
# patient_data.to_pickle("./patient_data.pkl")
# patient_data_test.to_pickle("./patient_data_test.pkl")
patient_data = pd.read_pickle("./patient_data.pkl")
patient_data_test = pd.read_pickle("./patient_data_test.pkl")
print(patient_data, patient_data_test)

feature_list = patient_data.columns.tolist()[3:]
print(feature_list)

#calculate averages to insert for nan's where there is no data
averages = train_features.mean(axis=0)


#replace nan's in the data: 
#could add step to drop patients with barely any non-nan values
#all nan's: replace all values with the average of all the data
# one not-nan value: replace all nan's with this value
# more than one not-nan value: linear regression on those values and replace nan's with the results of the fit.
np.seterr('raise')
for index, row in patient_data.iterrows():
    for jj in feature_list:
        number_not=[]
        number_nan=[]
        for kk in range(len(row[jj])):
            if not np.isnan(row[jj][kk]):
                number_not.append(kk)
            else:
                number_nan.append(kk)
        if number_not==[]:
            row[jj]=[averages[jj] for i in range(12)]
        elif len(number_not)==1:
            row[jj]=[row[jj][number_not[0]]for i in range(12)]
        else:
            x=[list(range(12))[i] for i in number_not]
            y=[row[jj][i] for i in number_not]
            slope, intercept, a,b,c =stats.linregress(x,y)
            for hh in number_nan:
                row[jj][hh]=intercept+slope*list(range(12))[hh]


###################test_data######################
#replace nan's in the data: 
#could add step to drop patients with barely any non-nan values
#all nan's: replace all values with the average of all the data
# one not-nan value: replace all nan's with this value
# more than one not-nan value: linear regression on those values and replace nan's with the results of the fit.
np.seterr('raise')
for index, row in patient_data_test.iterrows():
    for jj in feature_list:
        number_not=[]
        number_nan=[]
        for kk in range(len(row[jj])):
            if not np.isnan(row[jj][kk]):
                number_not.append(kk)
            else:
                number_nan.append(kk)
        if number_not==[]:
            row[jj]=[averages[jj] for i in range(12)]
        elif len(number_not)==1:
            row[jj]=[row[jj][number_not[0]]for i in range(12)]
        else:
            x=[list(range(12))[i] for i in number_not]
            y=[row[jj][i] for i in number_not]
            slope, intercept, a,b,c =stats.linregress(x,y)
            for hh in number_nan:
                row[jj][hh]=intercept+slope*list(range(12))[hh]


#create descriptor array - average and slope
X=[]
for index, row in patient_data.iterrows():
    dummy=[]
    dummy.append(row['Age'][0])
    for jj in feature_list:
        dummy.append(np.mean(row[jj]))
        dummy.append(np.min(row[jj]))
        dummy.append(np.max(row[jj]))
        dummy.append(row[jj][-1]-row[jj][0])
        dummy.append(np.std(row[jj]))
    X.append(dummy)
X=np.asarray(X)
print(X.shape)
np.isnan(X).any()



#################test-data###############
#create descriptor array - average and slope
X_test=[]
for index, row in patient_data_test.iterrows():
    dummy=[]
    dummy.append(row['Age'][0])
    for jj in feature_list:
        dummy.append(np.mean(row[jj]))
        dummy.append(np.min(row[jj]))
        dummy.append(np.max(row[jj]))
        dummy.append(row[jj][-1]-row[jj][0])
        dummy.append(np.std(row[jj]))
    X.append(dummy)
X=np.asarray(X)
print(X.shape)
np.isnan(X).any()



y_sepsis=train_labels['LABEL_Sepsis']
features_sepsis=[ 1,  1,  1,  2,  1,  1,  1,  1,  1,  6,  6,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  5,  1,  3, 12, 10,
        1,  1,  1,  7,  5,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1, 13, 10,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1, 11, 11,  1,  1,  1,  1,  1,  8,  9,  9, 14, 14,
        1,  1,  1,  4,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1, 13, 12,  1,  1,  1,  7,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1]
deletelist=[]
for ii in range(len(features_sepsis)):
    if features_sepsis[ii]>1:
        deletelist.append(ii)
X_sepsis=np.delete(X,deletelist,1)



brfc_sepsis=BalancedRandomForestClassifier(random_state=0,n_estimators = 400, max_depth = 100, max_features = 'sqrt',sampling_strategy=1,bootstrap=False).fit(X_sepsis,y_sepsis)



for ii in range(len(features_sepsis)):
    if features_sepsis[ii]>1:
        deletelist.append(ii)
X_test_sepsis=np.delete(X_test,deletelist,1)
y_pred_sepsis=brfc_sepsis.predict_proba(X_test_sepsis)[:,1]



y_pred_all=[]
y_pred_all.append(list(patient_data_test.keys()))



roc_score=[]
for ii in range(1,11):
    y=train_labels[train_labels.columns[ii]]
    brfc=BalancedRandomForestClassifier(random_state=0,n_estimators = 300, max_depth = 100, max_features = 'sqrt',sampling_strategy=1,bootstrap=False).fit(X,y)
    y_pred=brfc.predict_proba(X_test)[:,1]
    y_pred_all.append(y_pred)



y_pred_all.append(y_pred_sepsis)



feature=[[ 1, 32, 27, 32, 31, 38, 21, 14, 21, 38, 40, 13,  8, 15, 35, 40, 23,
        17, 20, 29, 35,  1, 12, 14,  1, 10, 16, 11, 11, 27, 33, 16, 23, 21,
        39, 41, 14, 24, 22, 25, 28,  1,  1,  1,  1,  1, 34, 31, 36, 42, 43,
        17, 20, 10, 39, 41,  4,  1,  1, 29, 38, 13, 18,  8, 37, 40,  4,  1,
         4, 10, 24, 28, 28, 31, 42, 45, 13,  8, 17, 22, 25,  1,  1,  5, 29,
        36, 16, 20, 24, 30, 33,  1,  1,  1,  7, 15,  1,  3,  3,  1,  1, 18,
         9, 18, 34, 39,  5,  5,  1, 23, 26,  1,  6,  7,  3,  1, 12,  6, 15,
        30, 36, 27, 26, 25, 44, 46,  1,  6, 12,  1,  1, 44, 43, 44, 46, 46,
        11, 19, 19, 35, 41,  7,  2,  1, 22, 26,  1,  1,  1,  1,  1, 32, 30,
        33, 45, 45, 37, 37, 34, 42, 43,  1,  1,  2,  1,  1,  9,  1,  9,  2,
        19],
 [ 1, 16, 17, 19, 22, 19,  2,  4,  2, 23, 24,  1,  1,  1, 17, 21, 11,
        10, 11, 21, 24,  1,  1,  1,  1,  1,  1,  1,  1, 15, 16,  3,  6,  7,
        25, 25,  9,  1,  3, 12,  1,  1,  1,  1,  1,  1, 21, 20, 23, 29, 30,
         1,  1,  1, 22, 25,  1,  1,  1, 15, 18,  1,  1,  1, 17, 19,  3,  4,
         1, 15,  8,  8,  6,  6, 27, 28,  5,  1,  1,  5,  1,  1,  1,  1, 16,
        18, 13,  7, 12, 22, 18,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1, 20, 23,  1,  1,  1,  9, 13,  1,  1,  1,  1,  1,  1,  1,  1,
        14, 12, 11, 10,  8, 28, 28,  1,  1,  1,  1,  1, 26, 27, 27, 30, 30,
         1,  1,  1, 20, 24,  1,  1,  1,  7,  9,  1,  1,  1,  1,  1, 10, 14,
        14, 29, 29,  2,  1,  1, 26, 26,  1,  1,  1,  1,  1,  5,  1,  4, 13,
         1],
 [ 2, 39, 34, 38, 43, 46, 23, 26, 23, 39, 43, 21, 14, 24, 41, 49,  5,
        14, 18, 36, 45,  1, 14, 12, 17, 24, 11, 16, 20, 42, 46,  7, 17,  1,
        46, 48, 23, 30, 26, 44, 44,  1, 11, 17,  4,  3, 47, 47, 45, 53, 53,
        25, 29, 26, 42, 49, 19,  9, 12, 40, 48, 21, 24, 19, 43, 48, 15, 28,
         7, 35, 35, 32, 32, 36, 50, 52, 20, 27, 22, 31, 37,  9,  9, 13, 44,
        45, 31, 28, 33, 42, 40,  4,  1, 12, 22, 27,  1, 10,  8,  3,  4, 27,
        19, 25, 38, 47, 16, 10, 18, 28, 35,  5, 15, 13, 13, 18,  6, 15, 20,
        30, 41, 32, 38, 34, 52, 54,  1,  1,  1,  1,  1, 51, 51, 52, 54, 54,
        29, 29, 22, 40, 49,  5,  2, 16, 33, 37,  1,  3,  6,  2,  6, 41, 36,
        39, 51, 53, 31, 34, 37, 50, 50,  1,  7, 10,  8,  8, 21, 11, 25, 30,
        33],
 [ 1, 39, 39, 33, 39, 36, 20, 18, 18, 38, 42, 11,  6, 15, 37, 41, 24,
        20, 21, 37, 35,  1, 11,  3,  5, 16, 15,  6, 15, 30, 35, 25, 21, 25,
        42, 44, 24, 26, 27, 35, 34,  1,  3,  2,  1,  1, 41, 44, 40, 48, 49,
        22, 20, 19, 43, 44,  4,  9,  1, 33, 38, 14,  4, 11, 38, 42, 19,  6,
        13, 30, 23, 28, 28, 28, 46, 47,  9,  2, 19, 30, 33,  5, 12, 12, 32,
        36, 29, 31, 26, 41, 43,  4,  1,  1, 12, 17,  1,  5,  7,  1,  1, 17,
        10, 18, 37, 40,  8, 13,  8, 24, 29,  1,  7, 10,  3,  9,  1,  7, 14,
        32, 36, 27, 25, 23, 47, 47,  1, 14, 16,  1,  1, 46, 46, 45, 49, 49,
        22, 23, 22, 40, 43,  8,  1, 10, 21, 26,  1,  1,  1,  1,  1, 32, 31,
        31, 48, 48, 34, 29, 34, 45, 45,  1,  1,  2,  1,  1, 13,  1, 16, 27,
        17]]



for ii in range(12,16):
    dum=0
    delete_list=[]
    for jj in range(len(feature[dum])):
        if feature[dum][jj]>1:
            deletelist.append(jj)
    X_new=np.delete(X,deletelist,1)
    X_test_new=np.delete(X_test,deletelist,1)
    dum+=1
    
    y=train_labels[train_labels.columns[ii]]
    rfr=RandomForestRegressor(random_state=0, n_estimators=300,max_depth=150, max_features=0.4).fit(X_new,y)
    y_pred=rfr.predict(X_test_new)
    y_pred_all.append(y_pred)



result=np.asarray(y_pred_all)
result=np.transpose(result)
print(result.shape)




df_result=pd.DataFrame(result,None,train_labels.columns)
df_result.to_csv('results.zip', index=False, float_format='%.3f', compression='zip')