import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, List
from util.env_vars import config
from util.dirs import check_path, create_dir, create_path


def save_fig(file_name: str, plot_format: str = "png") -> None:
    """Save a image given a name

    Args:
        file_name (str): Name of the file to be created
        plot_format (str, optional): Format to image to be created. Defaults to "png"
    """
    if not check_path(config["save"]):
        create_dir(config["save"])

    file_path = create_path(config["save"], f"{file_name}.{plot_format}")
    plt.savefig(file_path)
    plt.clf()


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
    plt.show()

    if "s" in kwargs or "save" in kwargs:
        save_fig(title + id)


def scatter(positions: List[pd.Series], title: str = "Scatter map", **kwargs: Any) -> None:
    """Creates a scatter plot given an array of series

    Args:
        positions (List[pd.Serries]): Positions for scatter plot (X, Y)
        title (str, optional): Title to show in the plot. Defaults to "Heat map".
    """

    id = kwargs.get("id") or kwargs.get("identifier") or ""
    color = kwargs.get("c") or kwargs.get("color") or "#ff7f0e"
    x_label = kwargs.get("xl") or kwargs.get("x_label") or ""
    y_label = kwargs.get("yl") or kwargs.get("y_label") or ""

    plt.scatter(positions[0], positions[1], c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

    if "s" in kwargs or "save" in kwargs:
        save_fig(title + id)
