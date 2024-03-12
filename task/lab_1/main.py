import time
from util.env_vars import config
from research.main import research_1


research = {
    "1": research_1,
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
