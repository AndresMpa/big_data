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

'''
To analyze a new data with previous data
'''
new_data = [8, 8]
k_means.predict([new_data])

for point in np.unique(label):
    plt.scatter(dataset[label == point, 0], dataset[label == point, 1], label = point)

'''
To plot a single data (If you need more simple use a for)
'''
plt.scatter(new_data[0], new_data[1], s=120, color="r", marker="^")

plt.scatter(centroids[:,0], centroids[:,1], s=80, color="k", marker="x")
plt.legend()
plt.show()

'''
Teacher notes:
- It's better to train with 80% of database
- It's better to predict with 20% of database
'''