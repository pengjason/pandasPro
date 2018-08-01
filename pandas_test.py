import os
import pandas as pd
import requests

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import numpy as np

PATH = r'D:\\Eclipse_worksplace\\pandasPro\\data\\'
  
# r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# 
# with open(PATH + 'iris.data','w') as f:
#     f.write(r.text)
#    
retval = os.getcwd()
print('current dir:%s'%retval)
os.chdir(PATH)

df = pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])

print(df.head())

print(df['sepal length'])
print(df.ix[:3,:2])
print(df.ix[:3,[x for x in df.columns if 'width' in x]])
print(df['class'].unique())

print(df[df['class']=='Iris-virginica'])
print(df.count())
print(df[df['class']=='Iris-virginica'].count())


virginica = df[df['class']=='Iris-virginica'].reset_index(drop=True)
virginica

df[(df['class']=='Iris-virginica')&(df['petal width']>2.2)]

df.describe()
df.describe(percentiles=[0.20,0.40,0.80,0.90,0.95])
df.corr()
df.corr(method="spearman")
# df.corr(method="kendall")

fig,ax = plt.subplots(figsize=(6,4))
ax.hist(df['petal width'],color='black')
ax.set_ylabel('Count',fontsize=12)
ax.set_xlabel('Width',fontsize=12)
plt.title('Iris Petal Width',fontsize=14,y=1.01)
retval = os.getcwd()
print('current dir:%s' %retval)
print('test')