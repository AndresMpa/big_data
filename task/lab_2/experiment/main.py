from pyspark.sql.functions import col, substring, when, regexp_replace, to_date
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql import SparkSession

import pyspark.ml.regression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from util.plotting import plt_regression, scatter


def experimental_case(df: SparkSession, timestamp: str):
    features = [col for col in df.columns]
    features = features[:2] + features[3:] + features[2:3]
    features = features[:3] + features[4:] + features[3:4]

    assembler = VectorAssembler(
        inputCols=features[:-1], outputCol="features")
    df = assembler.transform(df)
    df.show()

    lr = LinearRegression(featuresCol="features", labelCol="score")
    trainData, testData = df.randomSplit([0.9, 0.1])

    model = lr.fit(trainData)
    predictions = model.transform(testData)

    predctions = predictions.select("score", "prediction").toPandas()
    print(predctions["score"])
    print(predctions["prediction"])

    scatter(
        [predctions["score"], predctions["prediction"]],
        title="Actual vs Predicted values",
        x="Score",
        y="Prediction",
        id=f"Scatter - {timestamp}",
        s=True,
    )

    plt_regression(
        prediction=predctions["prediction"],
        features=predctions["score"],
        target=predctions["score"],
        id=f"Regression - {timestamp}",
        x="Score",
        y="Prediction",
        s=True,
    )
