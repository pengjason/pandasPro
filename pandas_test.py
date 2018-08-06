import os
import pandas as pd
import requests

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from pandas.tests.groupby.test_function import test_size
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

np.isnan
if np.isnan(df).any():
    print('has nan data')
else:
    print('data is good')
    
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
sns.violinplot(x=df['class'],y=df['sepal length'],ax=ax[0,0])
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

df['petal area'] = df.apply(lambda r : r['petal length']*r['petal width'],axis=1)
df

df.applymap(lambda v: np.log(v) if isinstance(v, float) else v)

df.groupby('class').mean()

df.groupby('class').describe()

df.groupby('petal width')['class'].unique().to_frame()

df.groupby('class')['petal width'].agg({'delta': lambda x: x.max() - x.min(),'max':np.max,'min':np.min})

fig,ax = plt.subplots(figsize=(7,7))
ax.scatter(df['sepal width'][:50],df['sepal length'][:50])
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs . Sepal Length',fontsize=14,y=1.02)

import statsmodels.api as sm
y = df['sepal length'][:50]
x = df['sepal width'][:50]
X = sm.add_constant(x)
results = sm.OLS(y,x).fit()
print(results.summary())

fig,ax = plt.subplots(figsize=(7,7))
ax.plot(x,results.fittedvalues,label='regression line')
ax.scatter(x,y,label='data point',color='r')
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs. Sepal Length',fontsize=14,y=1.02)
ax.legend(loc=2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

clf = RandomForestClassifier(max_depth=5,n_estimators=10)

X = df.ix[:,:4]
y = df.ix[:,4]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
clf.fit(X_train,y_train.astype(str))
y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['predicted','actual'])
rf['correct'] = rf.apply(lambda r: 1 if r['predicted']==r['actual'] else 0,axis=1)
rf

ratio = rf['correct'].sum()/rf['correct'].count()
print(ratio)
print(df)

f_importances = clf.feature_importances_
f_names = df.columns[:4]
f_std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)

zz = zip(f_importances,f_names,f_std)
zzs = sorted(zz,key=lambda x: x[0],reverse=True)
imps = [x[0] for x in zzs]
labels = [x[1] for x in zzs]
errs = [x[2] for x in zzs]
plt.bar(range(len(f_importances)),imps,color='r',yerr=errs,align='center')
plt.xticks(range(len(f_importances)),labels)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

clf = OneVsRestClassifier(SVC(kernel='linear'))

X = df.ix[:,:4]
y = np.array(df.ix[:,4]).astype(str)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['predicted','actual'])
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0,axis=1)
rf

rf['correct'].sum()/rf['correct'].count()


retval = os.getcwd()
print('current dir:%s' %retval)
print('test end')