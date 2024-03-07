'''
This is not the best but it pretty useful to achieve understanding
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset_a = np.array([
    [1], [1.5], [2], [2.5], [3.5]
])

dataset_b = np.array([
    [1.6], [2], [2.5], [1.5], [3]
])

plt.scatter(dataset_a, dataset_b)
# plt.show()

regression = LinearRegression(fit_intercept=True)
regression.fit(dataset_a, dataset_b)

plt.scatter(dataset_a, dataset_b, color="blue")
plt.plot(dataset_a, regression.predict(dataset_a), color="red", linewidth=3)
# plt.show()

# print(regression.coef_)
# print(regression.intercept_)

'''
The regression equation would be:

f(x) = 0.42972973 * X + 1.21756757 * Y

or

f(x) = regression.coef_ * X + regression.intercept_ * Y
'''

'''
Adding a prediction for a new data
'''

new_data = np.array([
    [2.4485]
])

plt.scatter(new_data, regression.predict(new_data), color="g", marker="^", s=100)
plt.scatter(dataset_a, dataset_b, color="blue")
plt.plot(dataset_a, regression.predict(dataset_a), color="red", linewidth=3)
plt.show()