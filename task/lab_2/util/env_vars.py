import os
from pathlib import Path
from dotenv import load_dotenv
from os.path import join, dirname


project_path = Path(dirname(__file__))
env_file = join(project_path.parent.absolute(), ".env")

load_dotenv(env_file)

config = {
    "stage": os.environ.get("STAGE"),

    "dataset_percentage": float(os.environ.get("DATASET_PERCENTAGE")),
    "dataset_dir": os.environ.get("DATASET_DIR"),
    "save": os.environ.get("SAVE_DIR"),

    "spark_cores": os.environ.get("SPARK_CORES"),
    "spark_shuffle": int(os.environ.get("SPAR_SHUFFLE")),
    "spark_exec_memory": os.environ.get("SPARK_EXEC_MEMORY"),
    "spark_drive_memory": os.environ.get("SPARK_DRIVE_MEMORY"),

    "verbose": os.environ.get("VERBOSE") == "1",
    "show_plot": os.environ.get("SHOW_PLOT") == "1",
}
