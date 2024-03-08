import pandas as pd
from typing import List

def clear_data(dataset: pd.DataFrame, columns_to_drop: List[int] = [], *args: bool) -> pd.DataFrame:
    """Clear data from a dataset

    Args:
        dataset (DataFrame): Receive a pandas DataFrame
        columns_to_drop (List[int]): A list of indexes of columns to remove from the dataset

    Returns:
        dataset: Same dataset without NaN or some columns when needed
    """
    dataset.drop(dataset.columns[columns_to_drop], axis=1, inplace=True)

    if args:
        print(f"{dataset.isna().sum()} NaN removing")
        
    dataset = dataset.fillna(0)
    return dataset