'''
Created on 2018. 1. 18.

@author: jaehyeong
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
import itertools
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH'] 


class_name = [0, 1]
def plot_confusion_matrix(model, X_test, y_test, classes,
                          normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def cf_report(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred)
    
    print(cm)
    print(report)

def ml(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(C=0.01)
    log.fit(X_train, y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    '''
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    param = {
        'max_depth' : 3,
        'eta' : 0.1,
        'learning_rate' : 0.01,
        'silent' : 0,
        'objective' : 'multi:softmax',
        'booster' : 'gbtree',
        'n_estimators' : 100,
        'num_class' : 2
        }
    
    bst = xgb.train(param, dtrain, num_boost_round=500)
    train_pred = bst.predict(dtrain)
    test_pred = bst.predict(dtest)
    
    from sklearn.metrics import accuracy_score
    print('train set score : ', accuracy_score(y_train, train_pred))
    print('test set score : ', accuracy_score(y_test, test_pred))
    '''
    
    models={log:'Logistic Regression', dt:'Decision Tree', rf:'Random Forest', svm:'SVM',mlp:'Multi layer perceptron'}
    return models

if __name__ == "__main__":
    df = pd.read_csv('../data/creditcard.csv')
    print(df.head())
    print(df.describe())
    
    # 칼럼간 정상/비정상 값의 분포가 유사한 칼럼은 제외
    df = df.drop(['V8','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis=1)
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    print('y : ',Counter(y))
    
    # under sampling
    from imblearn.under_sampling import RandomUnderSampler
    resampled = RandomUnderSampler(ratio=0.85, random_state=2018)
    X_res, y_res = resampled.fit_sample(X, y)
    print('y_res : ',Counter(y_res))
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X = scaler.transform(X_res)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_res, random_state=42)
    
    models = ml(X_train, y_train, X_test, y_test)
    
    for model in models.keys():
        print(models.get(model),' score : ',model.score(X_test, y_test))
        cf_report(model, X_test, y_test)

