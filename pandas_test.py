import os
import pandas as pd
import requests

import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline
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

fig,ax = plt.subplots(2,2,figsize=(6,4))
ax[0][0].hist(df['petal width'],color='black')
ax[0][0].set_ylabel('Count',fontsize=12)
ax[0][0].set_xlabel('Width',fontsize=12)
ax[0][0].set_title('Iris Petal Width',fontsize=14,y=1.01)

ax[0][1].hist(df['petal length'],color='black')
ax[0][1].set_ylabel('Count',fontsize=12)
ax[0][1].set_xlabel('Lenth',fontsize=12)
ax[0][1].set_title('Iris Petal Lenth',fontsize=14,y=1.01)

ax[1][0].hist(df['sepal width'],color='black')
ax[1][0].set_ylabel('Count',fontsize=12)
ax[1][0].set_xlabel('Width',fontsize=12)
ax[1][0].set_title('Iris Sepal Width',fontsize=14,y=1.01)

ax[1][1].hist(df['sepal length'],color='black')
ax[1][1].set_ylabel('Count',fontsize=12)
ax[1][1].set_xlabel('Length',fontsize=12)
ax[1][1].set_title('Iris Sepal Length',fontsize=14,y=1.01)

plt.tight_layout()

fig,ax = plt.subplots(figsize=(6,6))
ax.scatter(df['petal width'],df['petal length'],color='green')
ax.set_xlabel('Petal Width')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Scatterplot')

fig,ax = plt.subplots(figsize=(6,6))
ax.plot(df['petal length'],color='blue')
ax.set_xlabel('Specimen Number')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Length Plot')

fig,ax = plt.subplots(figsize=(6,6))
bar_width = 0.8
labels = [x for x in df.columns if 'length' in x or 'width' in x]
ver_y = [df[df['class']=='Iris-versicolor'][x].mean() for x in labels]
vir_y = [df[df['class']=='Iris-virginica'][x].mean() for x in labels]
set_y = [df[df['class']=='Iris-setosa'][x].mean() for x in labels]
x = np.arange(len(labels))
ax.bar(x,vir_y,bar_width,bottom=set_y,color='darkgrey')
ax.bar(x,set_y,bar_width,bottom=ver_y,color='white')
ax.bar(x,ver_y,bar_width,color='black')
ax.set_xticks(x+(bar_width/2))
ax.set_xticklabels(labels,rotation=-70,fontsize=12)
ax.set_title('Mean Feature Measurement By Class',y=1.01)
ax.legend(['Virginica','Setosa','Versicolor'])


import seaborn as sns
sns.pairplot(df,hue='class')


fig,ax = plt.subplots(2,2,figsize=(7,7))
sns.set(style='white',palette='muted')
sns.violinplot(x=dif['class'],y=df['sepal length'],ax=ax[0,0])
sns.violinplot(x=df['class'],y=df['sepal width'],ax=ax[0,1])
sns.violinplot(x=df['class'],y=df['petal length'],ax=ax[1,0])
sns.violinplot(x=df['class'],y=df['petal width'],ax=ax[1,1])
fig.suptitle('Violin Plots',fontsize=16,y=1.03)
for i in ax.flat:
    plt.setp(i.get_xticklabels(),rotation=-90)
fig.tight_layout()    

df['class'] = df['class'].map({'Iris-setosa':'SET','Iris-virginica':'VIR','Iris-versicolor':'VER'})
df

df['wide petal'] = df['petal width'].apply(lambda v: 1 if v>= 1.3 else 0)
df

df['petal area'] = df.apply(lambda r : r['petallength']*r['petal width'],axis=1)
df




retval = os.getcwd()
print('current dir:%s' %retval)
print('test')