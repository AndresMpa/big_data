"""
Neuronal Network Architecture
"""

from util.env_vars import config
from typing import Any, List, Tuple, Union, Optional

import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


ACTIVATION_FUNCTIONS = {
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential",
    "leaky_relu",
    "relu6",
    "silu",
    "hard_silu",
    "gelu",
    "hard_sigmoid",
    "linear",
    "mish",
    "log_softmax",
}


def create_nn(
    input_layer: List[int, int],
    hidden_layer: List[List[Union[int, str, Optional[int]]]],
    output_layer: int,
) -> Sequential:
    """Creates any simple neural network given a architecture as a list

    Args:
        input_layer (List[int, int]): Defines NN inputs as [units, input_shape]
        hidden_layer (List[List[Union[int, str, Optional[int]]]]): Defines hidden layers of NN as [[units, activation_function, dropout (Optionally)]]. The activation_function must be any of the following functions:
        "relu", "sigmoid", "softmax", "softplus","softsign", "tanh", "selu", "elu", "exponential", "leaky_relu", "relu6", "silu", "hard_silu", "gelu", "hard_sigmoid", "linear", "mish" or "log_softmax"
        output_layer (int): Define the output layer

    Raises:
        ValueError: Raised when not meeting the hidden layer criteria
    """

    nn_arch = []
    nn_arch.append(Dense(units=input_layer[0], input_shape=input_layer[1]))

    for layer in hidden_layer:
        if hidden_layer[1] not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Invalid activation: {hidden_layer[1]}. Valid activations are: {', '.join(ACTIVATION_FUNCTIONS)}"
            )

        nn_arch.append(Dense(units=layer[0], activation=layer[1]))

        if layer[2] != 0:
            nn_arch.append(Dropout(layer[2]))

    nn_arch.append(Dense(units=output_layer))

    neuronal_network = Sequential(nn_arch)
    neuronal_network.compile(optimizer=Adam(0.2), loss="mean_squared_error")

    return neuronal_network


def train_nn(
    neuronal_network: Sequential, data: pd.DataFrame, target: pd.Series, **kwargs: Any
) -> Tuple[Sequential, list, list] | Tuple[Sequential, list, list, list, list]:
    """Simple Linear Regression model

    Args:
        neuronal_network (Sequential): A neuronal network instance
        data (pd.DataFrame): pd.DataFrame properly created (NxM like)
        target (pd.Series): A simple series Nx1 like

    Returns:
        Sequential: Returns a trained Neuronal Network instance
        list: X_test from train_test_split
        list: y_test from train_test_split
        list: X_train from train_test_split
        list: y_train from train_test_split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        target,
        train_size=config["train_size"],
        test_size=config["test_size"],
        random_state=42,
    )

    epochs = kwargs.get("epochs") or kwargs.get("e") or 1000

    neuronal_network.fit(X_train, y_train, epochs=epochs, verbose=True)

    if "f" in kwargs or "full" in kwargs:
        return neuronal_network, X_test, y_test, X_train, y_train
    else:
        return neuronal_network, X_test, y_test
