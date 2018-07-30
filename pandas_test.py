import os
import pandas as pd
import requests


PATH = r'D:\\Eclipse_worksplace\\pandasPro\\data\\'

r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

with open(PATH + 'iris.data','w') as f:
    f.write(r.text)
   
retval = os.getcwd()
print('current dir:%s'%retval)
os.chdir(PATH)

df = pd.read_csv(PATH+'iris.data',names=['sepal lenth','sepal width','petal length','pettal with','class'])

print(df.head())

retval = os.getcwd()
print('current dir:%s' %retval)
print('test')