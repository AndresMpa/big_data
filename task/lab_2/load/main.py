from pyspark.sql.types import StructField, StringType, IntegerType, StructType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from util.env_vars import config


def get_dataset(spark: SparkSession, file: str, separator: str, encoding: str = "UTF-8", columns_to_drop: [] = []):
    path = config["dataset_dir"]
    sparkDataFrame = spark.read.csv(
        f"{path}/{file}", sep=separator, header=True, encoding=encoding).sample(withReplacement=False, fraction=config["dataset_percentage"], seed=42)
    sparkDataFrame = sparkDataFrame.drop(*columns_to_drop)

    return sparkDataFrame


def adjust_string_columns(df: SparkSession, columns_to_index: []):
    indexed_df = df
    indexers = {}

    for column in columns_to_index:
        # Verificar las categorías únicas presentes en la columna
        unique_categories = indexed_df.select(
            column).distinct().rdd.map(lambda r: r[0]).collect()

        # Inicializar StringIndexer con handleInvalid="keep"
        indexer = StringIndexer(
            inputCol=column, outputCol=column+"_indexed", handleInvalid="keep")

        # Ajustar y transformar el DataFrame
        indexed_df = indexer.fit(indexed_df).transform(indexed_df)

        # Almacenar el índice asociado a cada valor único en un diccionario
        indexers[column] = dict(enumerate(indexer.fit(df).labels))

    # Reemplazar los valores originales con los índices asignados
    for indexer in indexers:
        indexed_df = indexed_df.drop(indexer).\
            withColumnRenamed(indexer+"_indexed", indexer)

    return indexed_df, indexers


def adjust_num_columns(df: SparkSession, columns_to_convert: []):
    converted_df = df

    for column in columns_to_convert:
        converted_df = converted_df.withColumn(
            column, df[column].cast(DoubleType()))

    return converted_df
