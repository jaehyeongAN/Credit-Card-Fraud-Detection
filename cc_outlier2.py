'''
Created on 2018. 1. 31.

@author: jaehyeong
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import mglearn

def preprocessing(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    return X_pca

def kmeans():
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=2018)
    kmeans.fit_predict(X)
    y_pred = kmeans.labels_
    print(Counter(y_pred))
    
    plt.figure(figsize=(8,8))
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='Paired', s=60, edgecolors='white')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
                marker='*',c=[mglearn.cm2(0), mglearn.cm2(1)],s=60, linewidth=2, edgecolor='red')
    plt.show()
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y_pred)
    ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1])
    ax.set_title('pcomp 1')
    ax.set_title('pcomp 2')
    ax.set_title('pcomp 3')
    plt.show()
    '''
    '''
    res_df = pd.DataFrame(X_a)
    res_df['Class'] = y_a
    res_df['Cluster'] = y_pred
    #res_df.to_csv('credit_cluster_5.csv')

def outlier_detect():
    # outlier detect
    from sklearn.ensemble.iforest import IsolationForest
    clf1 = IsolationForest(n_estimators=100, random_state=2018, contamination=0.09)
    clf1.fit(X_pca)
    outlier1 = clf1.predict(X_pca)
    print('isolation forest\n : ',Counter(outlier1))    # 1:normal, -1:outlier
    
    from sklearn.neighbors import LocalOutlierFactor
    clf2 = LocalOutlierFactor(n_neighbors=100, contamination=0.09)
    outlier2 = clf2.fit_predict(X_pca)
    print('localoutlierfactor\n : ',Counter(outlier2))
    
    return clf1, outlier1, clf2, outlier2

def outlier_plot():
    # plot
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(18, 9))
    ax1.scatter(X[:,0], X[:,1], c=outlier1, cmap='Paired', s=60, edgecolors='white')
    ax1.set_title('IsolationForest')
    ax2.scatter(X[:,0], X[:,1], c=outlier2, cmap='Paired', s=60, edgecolors='white')
    ax2.set_title('LocalOutlierFactor')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=outlier2)
    ax.set_xlabel('pcomp 1')
    ax.set_ylabel('pcomp 2')
    ax.set_zlabel('pcomp 3')
    plt.show()

def eval():
    df = pd.DataFrame(X_a)
    df['Class'] = y_a
    df['outlier2'] = outlier2
    df['outlier2'] = df['outlier2'].replace({1: 1, -1: 0})    # 여기서 outlier는 정상 값
    
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
    print('confusion matrix\n', confusion_matrix(df['Class'], df['outlier2']))
    print('accuracy : ',accuracy_score(df['Class'], df['outlier2']))
    print('recall : ',recall_score(df['Class'], df['outlier2']))
    print('precision : ',precision_score(df['Class'], df['outlier2']))
    print('f1-score : ',f1_score(df['Class'], df['outlier2']))
    print(classification_report(df['Class'], df['outlier2']))
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/credit_outlier.csv')
    df = df[df['outlier'] == 1] # outlier인 데이터만 
    print(df.head())
    print(Counter(df['Class']))
    
    X_a = df.drop(['Class'], axis=1)
    y_a = df['Class']
    X = np.asarray(X_a)
    
    X_pca = preprocessing(X)
    clf1, outlier1, clf2, outlier2 = outlier_detect()
    outlier_plot()
    df = eval()
    #df.to_csv('../data/credit_outlier2.csv', index=False) 
    
    #kmeans()
