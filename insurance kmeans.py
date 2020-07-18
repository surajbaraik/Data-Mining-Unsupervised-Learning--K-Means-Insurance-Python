# -*- coding: utf-8 -*-
"""
"""

# -*- coding: utf-8 -*-
"""


"""

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
import seaborn as sns

insurance = pd.read_csv("D:\\data sets\\Insurance Dataset.csv")
insurance.describe()
insurance.mean()
insurance.var
sns.pairplot(insurance.iloc[:,:])

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm = norm_func(insurance.iloc[:,1:])


df_norm.head(20)  

k = list(range(2,24))
k
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("premiums paid");plt.xticks(k)

model=KMeans(n_clusters=6) 
model
model.fit(df_norm)

model.labels_ 
md=pd.Series(model.labels_)  
insurance['clust']=md  
md 
df_norm.head()


insurance.iloc[:,1:7].groupby(insurance.clust).mean()
