import pandas as pd
from util.env_vars import config
from pandas import read_csv


def load_dataset(file: str, encoding: str = "latin-1") -> pd.DataFrame:
    """Load a dataset from a dataset path given in .env file

    Args:
        file (str): The dataset name
        encoding (str, optional): Encoding from the dataset file. Default to "latin-1"

    Returns:
        pd.DataFrame: A loaded file as a pandas dataFrame
    """
    path = config["datasets"]
    dataset = read_csv(f"{path}/{file}", encoding=encoding)
    return dataset    