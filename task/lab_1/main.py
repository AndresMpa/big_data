import time
from util.env_vars import config
from research.preliminary import correlation
from research.stream_speechiness import predict


research = {
    "1": correlation,
    "2": predict,
}


def select_stage(stage):
    return research.get(stage, "Error")


if __name__ == '__main__':
    # To track data
    timestamp = time.time()

    research = select_stage(config["stage"])

    if research == "Error":
        raise Exception("There is an error on stage definition")

    research(timestamp)