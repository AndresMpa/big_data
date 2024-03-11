import pandas as pd
from load.main import load_dataset
from preprocessing.main import clear_data
from analysis.main import correlation


dataset = load_dataset("Popular_Spotify_Songs.csv")
dataset = clear_data(
    dataset, [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22]
)
dataset["streams"] = pd.to_numeric(dataset["streams"], errors="coerce")

"""
- Correlation BPS and Streams
- Correlation Artist count and Streams
- Correlation Danceability count and Energy
- Inverse correlation Speechiness count and Streams
- Correlation Danceability and Streams
"""

print(dataset)

print(correlation(dataset, [1, 1, 2, 2, 3, 3]))
