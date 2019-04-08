import pandas as pd 

df = pd.read_csv('../data/creditcard.csv')
print(df.head())

data1 = df[df['V10']<=-2.178]
data2 = data1[data1['V17']<=-11.209]
data3 = data2[data2['V12']<=-10.033]
data4 = data3[data3['V3']>-2.318]

print(data4['Class'].value_counts())