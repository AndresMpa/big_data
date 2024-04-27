import findspark
from pyspark.sql import SparkSession
from util.env_vars import config

findspark.init()


def create_session():
    spark = (
        SparkSession.builder.appName("SparkSession")
        .config("spark.master", f"local[{config['spark_cores']}]")
        .config("spark.executor.memory", f"{config['spark_exec_memory']}g")
        .config("spark.sql.shuffle.partitions", config['spark_shuffle'])
        .config("spark.driver.memory", f"{config['spark_drive_memory']}g")
        .getOrCreate()
    )

    return spark
