from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import pandas as pd 
import numpy as np
import os
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

credit_data = pd.read_csv('creditcard_cluster.csv', encoding='CP949')
credit_data = credit_data.drop(['Class','Outlier'], axis=1)

X = credit_data.drop(['Cluster'],axis=1)
y = credit_data['Cluster']

sc = MinMaxScaler()
sc.fit(X)
#X = sc.transform(X)

# label & one-hot encoding
e = LabelEncoder()
y = e.fit_transform(y.astype(str))
y_encode = np_utils.to_categorical(y)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y_encode, test_size=0.3, random_state=seed)

# model 
model = Sequential()
model.add(Dense(24, input_dim=30, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', 
				optimizer='adam',
				metrics=['accuracy'])


model.fit(X_train, y_train, epochs=500, batch_size=10)
test_pred = model.predict(X_test)
prd
print('Accuracy: ',model.evaluate(X_test, y_test)[1])
'''


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print('confusion matrix\n', confusion_matrix(y_pred=pred, y_true=y_test))
print('RandomForest Accuracy: ',rf.score(X_test, y_test))

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print('confusion matrix\n', confusion_matrix(y_pred=pred, y_true=y_test))
print('Decision Tree Accuracy: ',dt.score(X_test, y_test))

# Tree graph
export_graphviz(dt, out_file='tree.dot', feature_names=X.columns,
                class_names=['0','1','2','3'], filled=True, impurity=False)

with open("tree.dot", encoding='utf-8') as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph,encoding='utf-8')
dot.format='svg'
dot.render(filename='tree', view=True)