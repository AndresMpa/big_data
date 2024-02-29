import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

covid = pd.read_csv("../datasets/Casos_positivos_de_COVID-19_en_Colombia.csv")

# Return NaN .sum() sum them
#print(covid.isna().sum())

"""
    How to trait missing data:
        - Erase it
        - Replace them for the mean
        - Aproximate it (Calculus)
        - Predicte it (Calculus)
"""

"""
    Instead of drop NaN an option can be fillna to fill those
    value with something (0 in this case)
"""
covid["atención"] = covid["atención"].fillna(0)
# print(covid.isna().sum())

"""
    It's useful to find cities wrote in the bad way
"""
# print(pd.unique(covid["Ciudad de ubicación"]))

"""
    Useful to get some statistics about the data
"""
# print(covid.describe())

"""
    Useful to get a box model
"""
amount_age = ["Edad"]
plt.figure(figsize=(20, 9))
covid[amount_age].boxplot()
plt.title("Age variable")
# plt.show()


"""
    This is useful to remove the outliers
"""
# print(covid[covid["Edad"] > 92])

"""
    To make the analysis faster we can transform the data through the separation of the database to make a corelation analysis
"""
recovery_per_department = covid[covid["atención"] == "Recuperado"].groupby("Departamento o Distrito")["atención"].count()
# print(recovery_per_department)

death_per_department = covid[covid["atención"] == "Fallecido"].groupby("Departamento o Distrito")["atención"].count()
# print(death_per_department)

"""
    To get the mean of ages per department
"""
age_mean = covid.groupby("Departamento o Distrito").agg({"Edad": "mean"})
# print(age_mean)

"""
    You should realize about something "Departamento o Distrito" is not a target
    it's a refence variable, this is a variable that works as a pivot, it's a condition
    to make smaller datasets, Why? Basically because this variable groups the largest
    amount of data.
    
    The reference value works a focus point to split the data
    
    Remember the target of out analysis was:
        - Age
        - Amount of death
        - Amount of recovery
    
    Other possibilities could be "Ciudad" or "Pertenecia etnica".
    
    If you need a refence variable we just need to fulfill something, that variable
    mustn't have duplicated data. It must have direct and indirect relations with Other
    fields or data
"""

"""
    Let's make a corelation analysis, first we need to merge the data frames
"""
covid_cases = pd.merge(
    death_per_department,
    recovery_per_department,
    on="Departamento o Distrito",
    suffixes=("_fallecidos", "_recuperados"))

covid_cases["total"] = covid_cases["atención_fallecidos"] + covid_cases["atención_recuperados"]

covid_cases = pd.merge(
    age_mean,
    covid_cases,
    on="Departamento o Distrito")

# print(covid_cases) # Everything merged, cool

"""
    There's a cool corelation analysis from pandas this is
"""

# print(covid_cases.corr())

"""
    According to the heatmap, there's no a corelation between age, death and recovery
"""
sns.heatmap(covid_cases.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlación de los datos en mapa de calor")
# plt.show()