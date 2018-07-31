import os
import pandas as pd
import requests


PATH = r'D:\\Eclipse_worksplace\\pandasPro\\data\\'

# r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# 
# with open(PATH + 'iris.data','w') as f:
#     f.write(r.text)
#    
retval = os.getcwd()
print('current dir:%s'%retval)
os.chdir(PATH)

df = pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','pettal width','class'])

print(df.head())

print(df['sepal length'])
print(df.ix[:3,:2])
print(df.ix[:3,[x for x in df.columns if 'width' in x]])
print(df['class'].unique())

print(df[df['class']=='Iris-virginica'])
retval = os.getcwd()
print('current dir:%s' %retval)
print('test')