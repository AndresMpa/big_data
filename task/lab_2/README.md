## Introduction

This project analyses [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset?select=final_animedataset.csv) from kaggle

### Usage

Create a new virtual environment using:

```bash

python -m venv env
pip install -r ./requirements.txt
```

Then copy .env.example into a .env file, here's an example of usage

```bash
STAGE=1

DATASET_PERCENTAGE=1
DATASET_DIR="../../datasets/big_datasets/anime/"
SAVE_DIR="plots"

SPARK_CORES=4
SPAR_SHUFFLE=1
SPARK_EXEC_MEMORY=1
SPARK_DRIVE_MEMORY=1
SPARK_LOG_LEVEL="OFF"

VERBOSE=1
SHOW_PLOT=1
```

Available options for `SPARK_LOG_LEVEL`:
    - "ALL"
    - "DEBUG"
    - "INFO"
    - "WARN"
    - "ERROR"
    - "FATAL"
    - "OFF"
