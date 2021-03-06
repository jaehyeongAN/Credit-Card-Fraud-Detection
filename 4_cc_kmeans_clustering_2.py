# 이상 분류 대상
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import mglearn
from collections import Counter

credit_data = pd.read_csv('creditcard_out.csv')
credit_data =credit_data[credit_data['Outlier']==1]

X = credit_data.drop(['Class','Outlier'], axis=1)
y = credit_data['Class']
z = credit_data['Outlier']

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X)
X_sc = sc.transform(X)

'''from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_sc)
X_pca = pca.transform(X_sc)'''

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=0)
km.fit(X_sc)
pred = km.predict(X_sc)
print(Counter(pred))	# Counter({1: 349, 0: 150})

# 2D
'''plt.scatter(X_pca[:,0], X_pca[:,1], c=pred, cmap='Paired', s=60, edgecolors='white')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            marker='*',c=[mglearn.cm2(0), mglearn.cm2(1)],s=60, linewidth=2, edgecolor='red')
plt.xlabel('feature 0'); plt.ylabel('feature 1')
plt.show()'''

# 비교
from sklearn.metrics import confusion_matrix, classification_report
print('confusion_matrix\n', confusion_matrix(y_true=y, y_pred=pred))
print('classification_report\n', classification_report(y_true=y, y_pred=pred))

# save
save_df = pd.DataFrame(X, columns=X.columns)
save_df['Class'] = y
save_df['Outlier'] = z
save_df['Cluster2'] = pred
save_df.to_csv('creditcard_cluster2.csv', index=False)
