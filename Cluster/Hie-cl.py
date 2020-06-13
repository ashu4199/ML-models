import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('../Dataset/Mall_Customers.csv')
x=data.iloc[:,-2:].values

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,color='red',label='Cluster-1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,color='blue',label='Cluster-2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,color='green',label='Cluster-3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,color='cyan',label='Cluster-4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,color='magenta',label='Cluster-5')
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.legend()
plt.title('Clusters of customers')
plt.xlabel('Annual_income')
plt.ylabel('Spending Score 1-100')
plt.show()