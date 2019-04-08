import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

credit_data = pd.read_csv('../data/creditcard.csv')

X = credit_data.drop(['Class'], axis=1)
y = credit_data['Class']

# under sampling
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(ratio=0.65, random_state=0)
X_res, y_res = sampler.fit_sample(X, y)
print('Class : ',Counter(y_res))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X_res)
X_sc = sc.transform(X_res)

# outlier detection 
from sklearn.ensemble.iforest import IsolationForest
clf = IsolationForest(n_estimators=300, contamination=0.40, random_state=0)
clf.fit(X_sc)
pred_outlier = clf.predict(X_sc)
pred_outlier = pd.DataFrame(pred_outlier).replace({1:0, -1:1})

# 평가
from sklearn.metrics import confusion_matrix, classification_report
print('confusion matrix\n', confusion_matrix(pred_outlier, y_res))
print('classification_report\n', classification_report(pred_outlier, y_res))


save_df = pd.DataFrame(X_res, columns=X.columns)
save_df['Class'] = y_res
save_df['Outlier'] = pred_outlier 
#save_df.to_csv('creditcard_out.csv', index=False)
