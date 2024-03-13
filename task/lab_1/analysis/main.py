import pandas as pd

from typing import List
from analysis.normalization import simple_scaling, min_max, z_score
from analysis.regression import linear_regression, test_regression


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


def get_regression(
    regression_type: str, data: pd.DataFrame, target: pd.Series, **kwargs
) -> None:
    if regression_type == "linear":
        regression, x_test, y_test = linear_regression(data, target)

    if "test" in kwargs:
        test_regression(regression, x_test, y_test)

    return regression
