import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

health = pd.read_csv("../../datasets/correlacion_ejemplo.csv")
#print(health.describe())
'''
Describing we realize that every characteristic owns a difference scale, due to this
we need to normalize
'''

# Simple normalization
health["edad"] = health["edad"] / health["edad"].max()

# Min max
health["altura"] = (health["altura"] - health["altura"].min()) / (health["altura"].max() - health["altura"].min())
health["peso"] = (health["peso"] - health["peso"].min()) / (health["peso"].max() - health["peso"].min())

# F-score wouldn't work due to the value it returns [Negative values]

#print(health.corr())

sns.heatmap(health.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlación de los datos en mapa de calor")
plt.show()