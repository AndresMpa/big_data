import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression
from analysis.regression import linear_regression, test_regression
from analysis.normalization import simple_scaling, min_max, z_score


def get_correlation(dataset: pd.DataFrame, normalization: List[int]) -> pd.DataFrame:
    """Generates a simple correlation analysis using a normalization strategy

    Args:
        dataset (pd.DataFrame): A dataset as a DataFrame
        normalization (List[int]): A list of normalization option where 0 means no normalize, 1 simple scaling, 2 min max algorithm and 3 z-score algorithm

    Returns:
        pd.DataFrame: Returns a simple correlation using pd.DataFrame.corr()
    """
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
) -> LinearRegression:
    """Creates a regression model using some specific type ("Linear")

    Args:
        regression_type (str): Regression type: "linear"
        data (pd.DataFrame): A dataset [X, Y, ...] like
        target (pd.Series): pd.Series like
        test (bool, optional): Can print a simple test over the regression

    Returns:
        LinearRegression: Returns a linear regression instance
    """
    if regression_type == "linear":
        regression, x_test, y_test = linear_regression(data, target)

    if "test" in kwargs:
        test_regression(regression, x_test, y_test)

    return regression
