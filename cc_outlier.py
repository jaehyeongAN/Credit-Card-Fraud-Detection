'''
Created on 2018. 1. 30.

@author: jaehyeong
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import mglearn

def preprocessing(X):
    from imblearn.under_sampling import RandomUnderSampler
    sampler = RandomUnderSampler(ratio=0.85, random_state=2018)
    X_res, y_res = sampler.fit_sample(X, y)
    print('undersampling : ', Counter(y_res))
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(X_res)
    X_pca = pca.transform(X_res)
    
    return X_res, y_res, X_pca
    

def outlier_detect():
    # outlier detect
    from sklearn.ensemble.iforest import IsolationForest
    clf1 = IsolationForest(n_estimators=100, random_state=2018, contamination=0.45)
    clf1.fit(X_pca)
    outlier1 = clf1.predict(X_pca)
    outlier1 = pd.DataFrame(outlier1).replace({1:0, -1:1})
    print('isolation forest\n : ', Counter(outlier1))
    
    from sklearn.neighbors import LocalOutlierFactor
    clf2 = LocalOutlierFactor(n_neighbors=300, contamination=0.45)
    outlier2 = clf2.fit_predict(X_pca)
    outlier2 = pd.DataFrame(outlier2).replace({1:0, -1:1})
    print('localoutlierfactor\n : ', Counter(outlier2))
    
    return clf1, outlier1, clf2, outlier2

def outlier_plot():
     # score distribution
    anomaly_score = clf1.decision_function(X_pca)
    
    res_df = pd.DataFrame(X_res)
    res_df['Class'] = y_res
    res_df['Scores'] = anomaly_score
    
    avg_0 = res_df.loc[res_df.Class == 0]
    avg_1 = res_df.loc[res_df.Class == 1]
    
    normal = plt.hist(avg_0.Scores, 50)
    plt.xlabel('Score distribution')
    plt.ylabel('Frequency')
    plt.title("Distribution of isolation forest score for normal observation")
    plt.show()
    
    abnormal = plt.hist(avg_1.Scores, 50)
    plt.xlabel('Score distribution')
    plt.ylabel('Frequency')
    plt.title("Distribution of isolation forest score for abnormal observation")
    plt.show()
    
    # plot
    '''f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    ax1.scatter(X[:, 0], X[:, 1], c=outlier1, cmap='Paired', s=60, edgecolors='white')
    ax1.set_title('IsolationForest')
    ax2.scatter(X[:, 0], X[:, 1], c=outlier2, cmap='Paired', s=60, edgecolors='white')
    ax2.set_title('LocalOutlierFactor')
    plt.show()'''
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=outlier1)
    ax.set_xlabel('pcomp 1')
    ax.set_ylabel('pcomp 2')
    ax.set_zlabel('pcomp 3')
    plt.show()
    
def eval():
    save_df = pd.DataFrame(X_res)
    save_df['Class'] = y_res
    save_df['outlier'] = outlier2

    from sklearn.metrics import confusion_matrix, classification_report
    print('confusion matrix\n', confusion_matrix(save_df['Class'], save_df['outlier']))
    print(classification_report(save_df['Class'], save_df['outlier']))
    
    return save_df
    
    
if __name__ == "__main__":
    df = pd.read_csv('../data/creditcard.csv')
    #df = df.drop(['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis=1)
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    X = np.asarray(X)
    print(Counter(y))
    
    X_res, y_res, X_pca = preprocessing(X)
    clf1, outlier1, clf2, outlier2 = outlier_detect()
    outlier_plot()
    res_df = eval()
    # res_df.to_csv('../data/credit_outlier.csv', index=False)