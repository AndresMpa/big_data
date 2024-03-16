from load.main import load_dataset
from preprocessing.main import clear_data
from analysis.main import get_correlation
from analysis.plotting import heat, scatter


def deeper(timestamp: str) -> None:
    """To look for some relations on the dataset

    Args:
        timestamp (str): A simple id to track the test
    """

    dataset = load_dataset("Popular_Spotify_Songs.csv")
    unused = [
        0,  # Track name (str)
        1,  # Artist(s) name (str)
        # 2,  # Artist count
        # 3,  # Released year
        # 4,  # Released month
        # 5,  # Released day
        # 6,  # In Spotify playlist
        # 7,  # In Spotify charts
        # 8,  # Streams (str)
        # 9,  # In apple playlist
        # 10,  # In apple charts
        # 11,  # In Deezer playlists
        # 12,  # In Deezer charts
        # 13,  # In Shazam charts
        # 14,  # BPM
        15,  # Key (str)
        16,  # Mode (str)
        # 17,  # Danceability (%)
        # 18,  # Valence (%)
        # 19,  # Energy (%)
        # 20,  # Acousticness (%)
        # 21,  # Instrumentalness (%)
        # 22,  # Liveness (%)
        # 23,  # Speechiness (%)
    ]

    dataset = clear_data(
        dataset=dataset,
        columns_to_drop=unused,
        nn=["streams", "in_deezer_playlists", "in_shazam_charts"],
    )

    normalization_strategy = [
        # Track name (str)
        # Artist(s) name (str)
        1,  # Artist count
        1,  # Released year
        1,  # Released month
        1,  # Released day
        1,  # In Spotify playlist
        1,  # In Spotify charts
        1,  # Streams (str)
        1,  # In apple playlist
        1,  # In apple charts
        1,  # In Deezer playlists
        1,  # In Deezer charts
        1,  # In Shazam charts
        1,  # BPM
        # Key (str)
        # Mode (str)
        1,  # Danceability (%)
        1,  # Valence (%)
        1,  # Energy (%)
        1,  # Acousticness (%)
        1,  # Instrumentalness (%)
        1,  # Liveness (%)
        1,  # Speechiness (%)
    ]
    correlation = get_correlation(dataset.copy(deep=True), normalization_strategy)

    heat(
        correlation,
        title="Correlation heat map for all variables",
        id=f"{timestamp}",
        s=True,
    )

    for column in dataset.columns:
        label_name = column.replace("_", " ").replace("%", "").capitalize()
        scatter(
            [dataset[column], dataset["streams"]],
            f"Streams vs {label_name}",
            id=f"{timestamp}",
            x=f"{label_name}",
            y="Streams",
            s=True,
        )
