from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from util.plotting import plt_regression, scatter, plot_roc_curve
import matplotlib.pyplot as plt


def experimental_case(df: SparkSession, timestamp: str):
    # Ensamblar las características en una sola columna
    features = [col for col in df.columns]
    assembler = VectorAssembler(
        inputCols=features[1:], outputCol="features_vector")
    df = assembler.transform(df)

    # Dividir los datos en conjunto de entrenamiento y prueba (por ejemplo, 70% para entrenamiento y 30% para prueba)
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=123)

    # Inicializar el modelo de regresión lineal
    lr = LinearRegression(featuresCol="features_vector", labelCol="score")

    # Entrenar el modelo con los datos de entrenamiento
    lr_model = lr.fit(train_data)

    # Hacer predicciones sobre el conjunto de prueba
    predictions = lr_model.transform(test_data)

    # Convertir los datos de predicción a un DataFrame de pandas para graficar
    predictions_df = predictions.select("score", "prediction").toPandas()

    scatter(
        [predictions_df["score"], predictions_df["prediction"]],
        title=f"Predictions vs True values",
        x="True values",
        y="Predictions",
        s=True,
    )

    # Evaluar el rendimiento del modelo utilizando la métrica de evaluación adecuada
    evaluator = RegressionEvaluator(
        labelCol="score", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
