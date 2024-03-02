'''
Clustering / K-means

Non supervised algorithm used to reduce the amount of data (Variables),
it splits the data into sets to have several analysis

PCA

Allows you use a set of variable to make plots in order to express as 
much as possible in few vars
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Alcohol (%) vs Sugar (g)
dataset = np.array([[1, 2], [2, 3], [4, 0], [3, 9], [6, 2], [3, 5], [9, 4], [2, 10], [5, 7]])
print(dataset)

k_means = KMeans(n_clusters=2, random_state=0)

'''
Training depends on amount of data
'''
label = k_means.fit_predict(dataset)

'''
To get centroids
'''
centroids = k_means.cluster_centers_
print(centroids)

for point in np.unique(label):
    plt.scatter(dataset[label == point, 0], dataset[label == point, 1], label = point)

plt.scatter(centroids[:,0], centroids[:,1], s=80, color="k", marker="x")
plt.legend()
plt.show()