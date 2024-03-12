import pandas as pd
from load.main import load_dataset
from preprocessing.main import clear_data
from analysis.main import get_regression


def predict(timestamp: float) -> None:
    """A deep analysis of hypothesis "Correlation Speechiness count and
        Streams" to achieve a better understanding of that correlation, using
        a linear regression

        Aims to be research #2

    Args:
        timestamp (str): A simple id to track the test
    """

    dataset = load_dataset("Popular_Spotify_Songs.csv")
    unused = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    dataset = clear_data(dataset=dataset, columns_to_drop=unused)

    # Just fixing 'streams' column (It's str by default)
    dataset["streams"] = pd.to_numeric(dataset["streams"], errors="coerce")

    get_regression(regression_type="linear",
                   data=dataset["speechiness_%"], target=dataset["streams"])
