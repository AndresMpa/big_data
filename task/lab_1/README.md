# Laboratory #1

## Study case

This [dataset](https://www.kaggle.com/datasets/arnavvvvv/spotify-music/discussion) contains the top songs on Spotify. The data includes various features such as song duration, artist details, album information, and song popularity

### Dataset structure:

| Track name       | Artist(s) name                 | Artist count                               | Released year                   | Released month                   | Released day                                | In Spotify playlist                                 | In Spotify charts                               | Streams                            | In apple playlist                                       | In apple charts                                     | In Deezer playlists                                | In Deezer charts                               | In Shazam charts                               | BPM              | Key                                 | Mode                                                   | Danceability (%)                                                                       | Valence (%)                                                                     | Energy (%)                                    | Acousticness (%)                                                      | Instrumentalness (%)                     | Liveness (%)                          | Speechiness (%)                                 |
| ---------------- | ------------------------------ | ------------------------------------------ | ------------------------------- | -------------------------------- | ------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- | ---------------------------------- | ------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------- | ----------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------- | ----------------------------------------------- |
| Name of the song | Name of the artist of the song | Number of artists contributing to the song | Year when the song was released | Month when the song was released | Day of the month when the song was released | Number of Spotify playlists the song is included in | Presence and rank of the song on Spotify charts | Total number of streams on Spotify | Number of Apple Music playlists the song is included in | Presence and rank of the song on Apple Music charts | Number of Deeser playlists the song is included in | Presence and rank of the song on Deezer charts | Presence and rank of the song on Shazam charts | Beats Per Minute | Chords scales (From major to minor) | Rhythmic relationship between long and short durations | Measures how suitable a song is for dancing based on a combination of musical elements | A measure from 0 to 100 describing the musical positiveness conveyed by a track | Perceptible measure of intensity and activity | A confidence measure from 0.0 to 1.0 of whether the track is acoustic | Predicts whether a track contains vocals | Refers directly to reverberation time | Detects the presence of spoken words in a track |

### Constraints

- Dataset must be shorter than 500 rows of data
- Variables in the problem selected should not extend 5

### Methodology

### Steps

##### Data preprocessing

Since, I'm not interested in analyzing specific cases, but generality; I removed "Track name" and "Artist(s) name", I'm not getting statistics about music consumption, so I also removed "Released year", "Released month" and "Released day"; I'm not interested in platforms neither, so I also removed "In Spotify playlist", "In Spotify charts", "In apple playlist", "In apple charts", "In Deezer playlists", "In Deezer charts" and "In Shazam charts". I pretend to analyze metada along with Streams, but all metada here represents a bunch of information I don't want to use completely, so I'm also removing "Key", "Mode", "Valence (%)", "Acousticness (%)" "Instrumentalness (%)" and "Liveness (%)". Removing all those columns I get a subset from this dataset to analyze the following metrics:

- Streams: Total number of streams on Spotify
- BPM: Beats Per Minute
- Danceability (%): Measures how suitable a song is for dancing based on a combination of musical elements
- Energy (%): Perceptible measure of intensity and activity
- Speechiness (%): Detects the presence of spoken words in a track

Those characteristics are represent in a table as:

| Streams | BPM    | Danceability | Energy | Speechiness |
| ------- | ------ | ------------ | ------ | ----------- |
| Number  | Number | Float        | Float  | Float       |

##### Preliminary data analysis (Hypothesis)

The most important characteristics for this analysis is going to be "Streams", streams represents the amount of reproductions on Spotify so, it makes sense to analyze in deep this metric, as a performance-like metric, the closes in relations characteristics I have found while reading the information from each column would be the following hypothesis:

- There's a correlation between BPS and Streams
- There's a correlation between Artist count and Streams
- There's a correlation between Danceability count and Energy
- There's a correlation between Speechiness count and Streams
- There's a correlation between Danceability and Streams

> Note: Indeed, I'm a latin researcher so there most be some biases in those hypothesis, due to the region I live, but the idea is to analyze them

Analyzing correlations most of hypothesis were refused by the correlation matrix, even so; there seems to be a correlation between 'Streams' and 'Speechiness', according to the following figure

##### Machine learning strategy

##### Results
