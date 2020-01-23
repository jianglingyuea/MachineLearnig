import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.DataFrame([[1,2,3],[2,3,4],[3,4,5]], index=['a','b','c'], columns=['d','e','f'])
print(df)
print(df._info_axis)
dic1={'name':['小明','小红','幽鬼','敌法'],'age':[17,20,5,40],'gender':['男','女','女','男']}
df=pd.DataFrame(dic1)
print(df)
print(df._info_axis)
df = df.drop(axis=1, columns=['age'],index=[0,1])
print(df)
sns.set()
df = sns.load_dataset("iris")
sns.pairplot(df, hue="species", size=2.5)
plt.show()