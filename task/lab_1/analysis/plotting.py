import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, List
from util.env_vars import config
from util.dirs import check_path, create_dir, create_path


def save_figure(figure: plt.Figure, file_name: str, plot_format: str = "png") -> None:
    """Save a image given a name

    Args:
        figure (plt.Figure): Instance of figure.gcf()
        file_name (str): Name of the file to be created
        plot_format (str, optional): Format to image to be created. Defaults to "png"
    """
    if not check_path(config["save"]):
        create_dir(config["save"])

    file_path = create_path(config["save"], f"{file_name}.{plot_format}")
    figure.savefig(file_path)


def heat(dataset: pd.DataFrame, title: str = "Heat map", **kwargs: Any) -> None:
    """Create a heat map given a dataset

    Args:
        dataset (pd.DataFrame): Dataset with some information (At least 2 columns)
        title (str, optional): Title to show in the plot. Defaults to "Heat map".
    """
    color = kwargs.get("c") or kwargs.get("color") or "YlGnBu"
    id = kwargs.get("id") or kwargs.get("identifier") or ""

    sns.heatmap(dataset, annot=True, cmap=color)
    plt.title(title)

    if "s" in kwargs or "save" in kwargs:
        figure = plt.gcf()
        save_figure(figure, f"{title} - {id}")

    plt.show()
    plt.close()


def scatter(
    positions: List[pd.Series], title: str = "Scatter map", **kwargs: Any
) -> None:
    """Creates a scatter plot given an array of series

    Args:
        positions (List[pd.Series]): Positions for scatter plot (X, Y)
        title (str, optional): Title to show in the plot. Defaults to "Scatter map".
    """

    id = kwargs.get("id") or kwargs.get("identifier") or ""
    color = kwargs.get("c") or kwargs.get("color") or "#ff7f0e"
    x_label = kwargs.get("x") or kwargs.get("x_label") or ""
    y_label = kwargs.get("y") or kwargs.get("y_label") or ""

    plt.scatter(positions[0], positions[1], c=color)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)

    if "s" in kwargs or "save" in kwargs:
        figure = plt.gcf()
        save_figure(figure, f"{title} - {id}")

    plt.show()
    plt.close()


def plt_regression(
    prediction: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    title: str = "Regression plot",
    color: List[str] = ["red", "blue"],
    **kwargs: Any,
) -> None:
    """Plots a regression as a scatter plot with a simple plot

    Args:
        prediction (pd.DataFrame): Simple DataFrame from a regression like process
        features (pd.DataFrame): Base DataFrame taken for prediction creation
        target (pd.Series): A pd.Series of labels or targets
        title (str, optional): Simple title to name the plot. Defaults to "Regression plot".
        color (List[str], optional): 2 color for plots. Defaults to ["red", "blue"].
    """

    id = kwargs.get("id") or kwargs.get("identifier") or ""
    x_label = kwargs.get("x") or kwargs.get("x_label") or ""
    y_label = kwargs.get("y") or kwargs.get("y_label") or ""

    plt.scatter(features, target, c=color[0], label="Current")
    plt.plot(features, prediction, c=color[1], label="Prediction")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)
    plt.legend()

    if "s" in kwargs or "save" in kwargs:
        figure = plt.gcf()
        save_figure(figure, f"{title} - {id}")

    plt.show()
    plt.close()


def plot(
    positions: List[pd.DataFrame], title: str = "Simple plot map", **kwargs: Any
) -> None:
    """Creates a simple plot given an array of series

    Args:
        positions (List[pd.DataFrame]): Positions for plot (X, Y)
        title (str, optional): Title to show in the plot. Defaults to "Simple plot map".
    """

    id = kwargs.get("id") or kwargs.get("identifier") or ""
    color = kwargs.get("c") or kwargs.get("color") or "#ff7f0e"
    x_label = kwargs.get("x") or kwargs.get("x_label") or ""
    y_label = kwargs.get("y") or kwargs.get("y_label") or ""

    plt.plot(positions[0], positions[1], c=color)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)

    if "s" in kwargs or "save" in kwargs:
        figure = plt.gcf()
        save_figure(figure, f"{title} - {id}")

    plt.show()
    plt.close()
