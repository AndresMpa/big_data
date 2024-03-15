import pandas as pd
from typing import Any, List, Tuple, Union, Optional

from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression
from analysis.neuronal_network import create_nn, train_nn
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


def get_nn(
    input_layer: int,
    hidden_layer: List[List[Union[int, str, Optional[int]]]],
    output_layer: int,
    data: pd.DataFrame,
    target: pd.Series,
    epochs: int,
) -> Tuple[Sequential, Sequential, list, list, list, list]:
    """Create a trained neuronal network

    Args:
        input_layer (int): Defines NN inputs
        hidden_layer (List[List[Union[int, str, Optional[int]]]]): Defines hidden layers of NN as [[units, activation_function, dropout (Optionally)]]. The activation_function must be any of the following functions:
        "relu", "sigmoid", "softmax", "softplus","softsign", "tanh", "selu", "elu", "exponential", "leaky_relu", "relu6", "silu", "hard_silu", "gelu", "hard_sigmoid", "linear", "mish" or "log_softmax"
        output_layer (int): Define the output layer
        data (pd.DataFrame): A dataset as a pd.DataFrame
        target (pd.Series): A series of targets
        epochs (int): Training epochs

    Returns:
        Sequential: A trained neuronal network
        train_log: Neuronal network history
        list: x_test
        list: y_test
        list: x_train
        list: y_train
    """
    nn = create_nn(
        input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer
    )
    nn, train_log, x_test, y_test, x_train, y_train = train_nn(
        neuronal_network=nn, data=data, target=target, epochs=epochs
    )

    return nn, train_log, x_test, y_test, x_train, y_train
