import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

#separate e.g. FWLG in columns x1: F, x2: W, x3: L, x4: G
N=len(data.Sequence)
M=len(data_test.Sequence)

x1 = np.array(data.Sequence[0][0])
x2 = np.array(data.Sequence[0][1])
x3 = np.array(data.Sequence[0][2])
x4 = np.array(data.Sequence[0][3])

x1_t = np.array(data_test.Sequence[0][0])
x2_t = np.array(data_test.Sequence[0][1])
x3_t = np.array(data_test.Sequence[0][2])
x4_t = np.array(data_test.Sequence[0][3])

for i in range(1,N):
    
    x1 = np.append(x1,data.Sequence[i][0])
    x2 = np.append(x2,data.Sequence[i][1])
    x3 = np.append(x3,data.Sequence[i][2])
    x4 = np.append(x4,data.Sequence[i][3])

for j in range(1,M):
    
    x1_t = np.append(x1_t,data_test.Sequence[j][0])
    x2_t = np.append(x2_t,data_test.Sequence[j][1])
    x3_t = np.append(x3_t,data_test.Sequence[j][2])
    x4_t = np.append(x4_t,data_test.Sequence[j][3])
    
y = data.Active.to_numpy()

#put data back together

numpy_data = np.array([x1,x2,x3,x4,y])
numpy_data_test = np.array([x1_t,x2_t,x3_t,x4_t])


df = pd.DataFrame(data=numpy_data, index=["x1","x2","x3","x4","y"])
df = df.transpose()

df_t = pd.DataFrame(data=numpy_data_test, index=["x1","x2","x3","x4"])
df_t = df_t.transpose()

#train data
X=df.drop('y',axis='columns')
#test data
X_t=df_t

#OneHotEncoder
transformer = make_column_transformer((OneHotEncoder(),['x1','x2','x3','x4']))

#first trial: Random Forest => not good enough results

rfclassifier = RandomForestClassifier(class_weight={1:20},n_estimators=200)
pipe = Pipeline([('transformer', transformer), ('c', rfclassifier)])

#final solution: MLPClassifier => good results

finalclassifier = MLPClassifier()
#RandomForestClassifier(class_weight={1:20},n_estimators=200,max_depth=50,max_features=0.1)
finalpipe=make_pipeline(transformer,finalclassifier)

finalpipe.fit(X,y)
y_t = finalpipe.predict(X_t)

for i in y_t:
    print(y_t[i])