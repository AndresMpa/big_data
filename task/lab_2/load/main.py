from pyspark.sql import SparkSession
from util.env_vars import config


def get_dataset(spark: SparkSession, file: str, separator: str, encoding: str = "UTF-8"):
    path = config["datasets"]
    sparkDataFrame = spark.read.csv(f"{path}/{file}", sep=separator, header=True, encoding=encoding)

    return sparkDataFrame