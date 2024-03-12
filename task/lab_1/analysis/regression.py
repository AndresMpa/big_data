import pandas as pd
from util.env_vars import config
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def linear_regression(data: pd.DataFrame, target: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, train_size=config["train_size"],
        test_size=config["test_size"], random_state=42)

    regression = LinearRegression(fit_intercept=True)
    regression.fit(X_train, y_train)

    return regression, X_test, y_test


def test_regression(regression, x_test, y_test) -> float:
    prediction = regression.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print(f'Algorithm accuracy was: {score:.2f}')

    return prediction
