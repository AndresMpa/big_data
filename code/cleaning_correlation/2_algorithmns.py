import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

covid = pd.read_csv("../datasets/Casos_positivos_de_COVID-19_en_Colombia.csv")

"""
    Let's check some normalization calculus

    Note: Each algorithm calculate data in a different way so you should use that
    data According to the data base instead of your expectative
"""

# Simple data scaling

simple_data_scaling = covid["Edad"] / covid["Edad"].max()

# Min - Max (Normalization 0, 1)

min_max = (covid["Edad"] - covid["Edad"].min()) / (covid["Edad"].max() - covid["Edad"].min())

# Z-score -> Uses standard deviation

z_core = (covid["Edad"] - covid["Edad"].mean()) / (covid["Edad"].std())

print(simple_data_scaling)
print(min_max)
print(z_core)