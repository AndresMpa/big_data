import pandas as pd

from typing import List
from analysis.normalization import simple_scaling, min_max, z_score


def get_correlation(dataset: pd.DataFrame, normalization: List[int]) -> pd.DataFrame:
    for index, column in enumerate(dataset):
        if normalization[index] == 0:
            dataset[column] = dataset[column]
        if normalization[index] == 1:
            dataset[column] = simple_scaling(dataset, column)
        if normalization[index] == 2:
            dataset[column] = min_max(dataset, column)
        if normalization[index] == 3:
            dataset[column] = z_score(dataset, column)

    return dataset.corr()
