from load.main import load_dataset
from preprocessing.main import clear_data
from analysis.main import get_nn
from analysis.plotting import plot, scatter, residual, histogram
from analysis.regression import test_regression


def nn_analysis(timestamp: str) -> None:
    """To look for some relations on the dataset

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
        # 5,  # Released day
        # 6,  # In Spotify playlist
        # 7,  # In Spotify charts
        # 8,  # Streams (str)
        # 9,  # In apple playlist
        # 10,  # In apple charts
        # 11,  # In Deezer playlists
        # 12,  # In Deezer charts
        # 13,  # In Shazam charts
        14,  # BPM
        15,  # Key (str)
        16,  # Mode (str)
        17,  # Danceability (%)
        18,  # Valence (%)
        19,  # Energy (%)
        20,  # Acousticness (%)
        21,  # Instrumentalness (%)
        22,  # Liveness (%)
        23,  # Speechiness (%)
    ]

    dataset = clear_data(
        dataset=dataset,
        columns_to_drop=unused,
        nn=["streams", "in_deezer_playlists", "in_shazam_charts"],
    )

    goal = dataset["streams"]
    narrows = dataset.drop("streams", axis=1)

    architectures = [
        # linear - 16 x 16 x 16
        {
            "input": [16, 8],
            "hidden": [[16, "linear", 0.3], [16, "linear", 0.3]],
            "output": 1,
            "epochs": 5,
            "arch": "linear - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "linear", 0.3], [16, "linear", 0.3]],
            "output": 1,
            "epochs": 1_000,
            "arch": "linear - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "linear", 0.3], [16, "linear", 0.2]],
            "output": 1,
            "epochs": 10_000,
            "arch": "linear - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "linear", 0.2], [16, "linear", 0.1]],
            "output": 1,
            "epochs": 100_000,
            "arch": "linear - 16 x 16 x 16",
        },
        # ReLu - 16 x 16 x 16
        {
            "input": [16, 8],
            "hidden": [[16, "relu", 0.3], [16, "relu", 0.3]],
            "output": 1,
            "epochs": 5,
            "arch": "relu - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "relu", 0.3], [16, "relu", 0.3]],
            "output": 1,
            "epochs": 1_000,
            "arch": "relu - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "relu", 0.3], [16, "relu", 0.2]],
            "output": 1,
            "epochs": 10_000,
            "arch": "relu - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "relu", 0.2], [16, "relu", 0.1]],
            "output": 1,
            "epochs": 100_000,
            "arch": "relu - 16 x 16 x 16",
        },
        # mish - 16 x 16 x 16
        {
            "input": [16, 8],
            "hidden": [[16, "mish", 0.3], [16, "mish", 0.3]],
            "output": 1,
            "epochs": 5,
            "arch": "mish - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "mish", 0.3], [16, "mish", 0.3]],
            "output": 1,
            "epochs": 1_000,
            "arch": "mish - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "mish", 0.3], [16, "mish", 0.2]],
            "output": 1,
            "epochs": 10_000,
            "arch": "mish - 16 x 16 x 16",
        },
        {
            "input": [16, 8],
            "hidden": [[16, "mish", 0.2], [16, "mish", 0.1]],
            "output": 1,
            "epochs": 100_000,
            "arch": "mish - 16 x 16 x 16",
        },
    ]

    for nn in architectures:
        neuronal_network, train_log, x_test, y_test, _, _ = get_nn(
            nn["input"], nn["hidden"], nn["output"], narrows, goal, epochs=nn["epochs"]
        )

        plot(
            train_log.history["loss"],
            title=f"Loss through {nn['epochs']} epochs",
            id=f"nn_analysis - {nn['arch']} {nn['epochs']} {timestamp}",
            s=True,
        )
        try:
            test_regression(neuronal_network, x_test=x_test, y_test=y_test)
        except:
            print("Error while testing")

        y_pred = neuronal_network.predict(x_test)
        residuals = y_test.values - y_pred.flatten()

        scatter(
            [y_test, y_pred],
            title="Actual vs Predicted values",
            x="Actual value",
            y="Predicted value",
            id=f"nn_analysis - {nn['arch']} {nn['epochs']} {timestamp}",
            s=True,
        )

        residual(
            [y_pred, residuals],
            title="Predicted values vs Residuals",
            x="Predicted value",
            y="Residual value",
            id=f"nn_analysis - {nn['arch']} {nn['epochs']} {timestamp}",
            s=True,
        )

        histogram(
            residuals,
            title="Residuals histogram",
            x="Residuals",
            y="Frequency",
            id=f"nn_analysis - {nn['arch']} {nn['epochs']} {timestamp}",
            s=True,
        )
