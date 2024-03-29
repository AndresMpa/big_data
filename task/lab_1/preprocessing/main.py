import pandas as pd
from typing import List, Any


def clear_data(
    dataset: pd.DataFrame, columns_to_drop: List[int] = [], *args: bool, **kwargs: Any
) -> pd.DataFrame:
    """Clear data from a dataset

    Args:
        dataset (DataFrame): Receive a pandas DataFrame
        columns_to_drop (List[int], optional): A list of indexes of columns to remove from the dataset. Defaults to []

    Returns:
        dataset: Same dataset without NaN or some columns when needed
    """
    no_numeric = kwargs.get("no_numeric") or kwargs.get("nn")
    for column in no_numeric:
        # Just fixing columns that are str by default
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset.drop(dataset.columns[columns_to_drop], axis=1, inplace=True)

    if args:
        print(f"{dataset.isna().sum()} NaN removing")

    dataset = dataset.fillna(0)
    return dataset


def get_features(data: pd.DataFrame, label: pd.Series):
    """Combine data and target into one DataFrame to split the data into features (X) and target (y)

    Args:
        data (pd.DataFrame): A DataFrame of information to be casted into features
        label (pd.Series): A Series of label that will the be target

    Returns:
        features: A DataFrame like return
        target: A Series of targets
    """
    df = pd.concat([data, label], axis=1)

    features = df[[data.name]]
    target = df[label.name]

    return features, target
