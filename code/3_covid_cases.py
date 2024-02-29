import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

covid = pd.read_csv("../datasets/Casos_positivos_de_COVID-19_en_Colombia.csv")
covid["atención"] = covid["atención"].fillna(0)

recovery_per_department = covid[covid["atención"] == "Recuperado"].groupby("Departamento o Distrito")["atención"].count()
# print(recovery_per_department)

death_per_department = covid[covid["atención"] == "Fallecido"].groupby("Departamento o Distrito")["atención"].count()
# print(death_per_department)

age_mean = covid.groupby("Departamento o Distrito").agg({"Edad": "mean"})
# print(age_mean)

"""
    Merging
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
# print(covid_cases)

covid_cases["atención_fallecidos"] = covid_cases["atención_fallecidos"] / covid_cases["atención_fallecidos"].max()
covid_cases["atención_recuperados"] = (covid_cases["atención_recuperados"] - covid_cases["atención_recuperados"].min()) / (covid_cases["atención_recuperados"].max() - covid_cases["atención_recuperados"].min())
covid_cases["Edad"] = (covid_cases["Edad"] - covid_cases["Edad"].mean()) / (covid_cases["Edad"].std())

sns.heatmap(covid_cases.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlación de los datos en mapa de calor")
plt.show()