from load.main import load_dataset
from preprocessing.main import clear_data, get_features
from analysis.plotting import plt_regression
from analysis.main import get_regression


def predict(timestamp: str) -> None:
    """A deep analysis of hypothesis "Correlation Speechiness count and
        Streams" to achieve a better understanding of that correlation, using
        a linear regression

        Aims to be research #2

    Args:
        timestamp (str): A simple id to track the test
    """

    dataset = load_dataset("Popular_Spotify_Songs.csv")
    unused = [
        0,  # Track name (str)
        1,  # Artist(s) name (str)
        2,  # Artist count
        3,  # Released year
        4,  # Released month
        5,  # Released day
        6,  # In Spotify playlist
        7,  # In Spotify charts
        # 8,  # Streams (str)
        9,  # In apple playlist
        10,  # In apple charts
        11,  # In Deezer playlists
        12,  # In Deezer charts
        13,  # In Shazam charts
        14,  # BPM
        15,  # Key (str)
        16,  # Mode (str)
        17,  # Danceability (%)
        18,  # Valence (%)
        19,  # Energy (%)
        20,  # Acousticness (%)
        21,  # Instrumentalness (%)
        22,  # Liveness (%)
        # 23,  # Speechiness (%)
    ]
    dataset = clear_data(dataset=dataset, columns_to_drop=unused, nn=["streams"])
    features, target = get_features(dataset["speechiness_%"], dataset["streams"])

    linear_regression = get_regression(
        regression_type="linear", data=features, target=target, test=True
    )

    plt_regression(
        linear_regression.predict(features),
        features,
        target,
        id=f"stream_speechiness - {timestamp}",
        x="Speechiness",
        y="Streams",
        s=True,
    )
