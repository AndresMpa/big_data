import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Temperature conversion:

Inputs:
- C°

Output:
- F°

Target: Cast F° into C°
'''

# Dataset
celsius = np.array([
    -40, -10, 0,  8, 15, 22,  38,  25.7, -10.2,  18.9, 30.5,   0.3,  12.8,  37.1,  -5.9,   8.6, 22
], dtype=float)

fahrenheit = np.array([
    -40, 14, 32, 46, 59, 72, 100, 78.26, 13.64, 66.02, 86.9, 32.54, 55.04, 98.78, 21.38, 47.48, 71.6
], dtype=float)

# Neuronal Network (NN)

'''
units = layers
input_shapes = neurons
'''
# tf.keras.layers.Dense
layers = Dense(units=1, input_shape=[1])

# tf.keras.Sequential
model = Sequential([layers])

model.compile(optimizer=tf.keras.optimizers.Adam(0.2), loss="mean_squared_error")

history = model.fit(celsius, fahrenheit, epochs=1000, verbose=True)

new_data = [-23]

result = model.predict(new_data)

'''
This is equivalent to:

(C° * 9/5) + 32 = F°
'''
print(f"Conversion for {new_data[0]} °C is {result[0][0]} °F")

'''
To get the weights from NN
'''
print(layers.get_weights())