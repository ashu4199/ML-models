import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv('../Dataset/Mall_Customers.csv')

x=data.iloc[:,-2:].values

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5,init ='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

# print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster-1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='Cluster-2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='Cluster-3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='cyan',label='Cluster-4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,color='magenta',label='Cluster-5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.legend()
plt.title('Clusters of customers')
plt.xlabel('Annual_income')
plt.ylabel('Spending Score 1-100')
plt.show()