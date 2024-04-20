import findspark
from pyspark.sql import SparkSession

findspark.init()


def create_session():
    spark = (
        SparkSession.builder.appName("SparkSession")
        .config("spark.master", "local[4]")
        .config("spark.executor.memory", "1g")
        .config("spark.sql.shuffle.partitions", 1)
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    print(spark)

    return spark

