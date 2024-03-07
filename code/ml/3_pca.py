'''
This is essentially principal components analysis

PCA reduces the amount of variables, representing behavior of previous database (Around 90 - 95% of database)

This is useful only to get a graphic
'''

'''
This is specifically quantitative
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


dataset = ([
    [1, -1, 2,  4,  7],
    [2,  9, 4,  3,  2],
    [7,  2, 5,  7,  8],
    [1, -3, 5, -2,  1],
    [2, -4, 3,  6, -5],
    [5, -4, 7, -2, -1]
])

'''
Number of vars = 5
Number of dimensions = 5
vars = dimensions
'''

'''
n_components define the amount of dimensions to reduce

In this examples it goes from 5 to 3
'''
pca = PCA(n_components=3)
pca.fit(dataset)

'''
explained_variance_ratio_ is used to get the variance
'''
print(pca.explained_variance_ratio_)

'''
Plotting you can see that actually around 90% of the database is being represented
'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Principal component")
plt.ylabel("Explained variance")
plt.show()

'''
To get PCA components matrix
'''
pca_matrix = pca.components_
print(pca_matrix)