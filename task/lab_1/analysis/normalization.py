import pandas as pd


def simple_scaling(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Simple data scaling normalization

    Args:
        data (pd.DataFrame): A pandas DataFrame
        target (str): The specific column in which the normalization should take place

    Returns:
        pd.DataFrame: The entire DataFrame with the normalization in <target>
    """
    data[target] = data[target] / data[target].max()
    return data


def min_max(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Min max normalization (Normalization 0, 1)

    Args:
        data (pd.DataFrame): A pandas DataFrame
        target (str): The specific column in which the normalization should take place

    Returns:
        pd.DataFrame: The entire DataFrame with the normalization in <target>
    """
    data[target] = (data[target] - data[target].min()) / (
        data[target].max() - data[target].min()
    )
    return data


def z_score(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Z-score normalization (Uses standard deviation)

    Args:
        data (pd.DataFrame): A pandas DataFrame
        target (str): The specific column in which the normalization should take place

    Returns:
        pd.DataFrame: The entire DataFrame with the normalization in <target>
    """
    data[target] = (data[target] - data[target].mean()) / (data[target].std())
    return data
