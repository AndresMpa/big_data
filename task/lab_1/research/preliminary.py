from load.main import load_dataset
from preprocessing.main import clear_data
from analysis.main import get_correlation
from analysis.plotting import heat, scatter


def correlation(timestamp: float) -> None:
    """To analyze first set of hypothesis:

        - Correlation BPM and Streams
        - Correlation Artist count and Streams
        - Correlation Speechiness count and Streams
        - Correlation Danceability and Streams
        - Correlation Danceability count and Energy

        Aims to be research #1

    Args:
        timestamp (str): A simple id to track the test
    """

    dataset = load_dataset("Popular_Spotify_Songs.csv")
    unused = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22]
    dataset = clear_data(dataset=dataset, columns_to_drop=unused)

    correlation = get_correlation(dataset.copy(deep=True), [1, 1, 1, 1, 1, 1])

    heat(correlation, title="Correlation heat map", id=f"{timestamp}", s=True)

    scatter(
        [dataset["bpm"], dataset["streams"]],
        "Streams vs BPM",
        id=f"{timestamp}",
        x="BPM",
        y="Streams",
        s=True,
    )
    scatter(
        [dataset["artist_count"], dataset["streams"]],
        "Streams vs Artiste count",
        id=f"{timestamp}",
        x="Artist count",
        y="Streams",
        s=True,
    )
    scatter(
        [dataset["speechiness_%"], dataset["streams"]],
        "Streams vs Speechiness (%)",
        id=f"{timestamp}",
        x="Speechiness",
        y="Streams",
        s=True,
    )
    scatter(
        [dataset["danceability_%"], dataset["streams"]],
        "Streams vs Danceability (%)",
        id=f"{timestamp}",
        x="Danceability",
        y="Streams",
        s=True,
    )
    scatter(
        [dataset["danceability_%"], dataset["energy_%"]],
        "Energy (%) vs Danceability (%)",
        id=f"{timestamp}",
        x="Danceability",
        y="Energy",
        s=True,
    )
