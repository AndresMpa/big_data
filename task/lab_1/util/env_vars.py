import os
from pathlib import Path
from dotenv import load_dotenv
from os.path import join, dirname


project_path = Path(dirname(__file__))
env_file = join(project_path.parent.absolute(), ".env")

load_dotenv(env_file)

config = {
    "datasets": os.environ.get("DATASET_DIR"),
    "save": os.environ.get("SAVE_DIR"),
    "stage": os.environ.get("STAGE"),
    "train_size": float(os.environ.get("TRAIN_SIZE")),
    "test_size": float(os.environ.get("TEST_SIZE")),
}
