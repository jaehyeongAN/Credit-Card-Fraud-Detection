'''
Created on 2018. 1. 18.

@author: jaehyeong
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

df = pd.read_csv('../data/creditcard.csv')
print(df.head())
print(df.describe())

# 시간대별 트랜잭션 양
f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))
ax1.hist(df.Time[df.Class==1], bins=50)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class==0], bins=50)
ax2.set_title('Normal')

plt.xlabel('Time(in Seconds)'); plt.ylabel('Number of Transactions')
plt.show()

# 금액대별 트랜잭션 양
f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))
ax1.hist(df.Amount[df.Class==1], bins=30)
ax1.set_title('Fraud')

ax2.hist(df.Amount[df.Class==0], bins=30)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()

# 정상/비정산 럼간 값 분포
v_features = df.ix[:,1:29].columns
for cnt, col in enumerate(df[v_features]):
    sns.distplot(df[col][df.Class==1], bins=50)
    sns.distplot(df[col][df.Class==0], bins=50)
    plt.legend(['Y','N'], loc='best')
    plt.title('histogram of feature '+str(col))
    plt.show()
