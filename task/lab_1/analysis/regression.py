import pandas as pd
from typing import Tuple, List
from util.env_vars import config
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def linear_regression(
    data: pd.DataFrame, target: pd.Series
) -> Tuple[LinearRegression, list, list]:
    """Simple Linear Regression model

    Args:
        data (pd.DataFrame): pd.DataFrame properly created for regressions NxM like
        target (pd.Series): A simple series Nx1 like

    Returns:
        LinearRegression: Returns a trained LinearRegression instance
        list: X_test from train_test_split
        list: y_test from train_test_split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        target,
        train_size=config["train_size"],
        test_size=config["test_size"],
        random_state=42,
    )

    regression = LinearRegression(fit_intercept=True)
    regression.fit(X_train, y_train)

    return regression, X_test, y_test


def test_regression(
    regression: RegressorMixin, x_test: List[float], y_test: List[float]
) -> float:
    """Test a regression using MAE, MSE and R^2

    Args:
        regression (RegressorMixin): A continues like regression (exa: LinearRegression)
        x_test (List[float]): A testable list (Typically from x_test)
        y_test (List[float]): A testable list (Typically from y_test)

    Returns:
        float: Prediction from regression.predict(x_test)
    """
    prediction = regression.predict(x_test)

    mae = mean_absolute_error(y_test, prediction)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    mse = mean_squared_error(y_test, prediction)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    r2 = r2_score(y_test, prediction)
    print(f"R-squared (R2): {r2:.2f}")

    return prediction


def test_classifier(
    classifier: ClassifierMixin, x_test: List[float], y_test: List[int]
) -> float:
    """Creates a simple test over a classifier like regression, measuring accuracy, precision, recall and F1-score

    Args:
        classifier (ClassifierMixin): A classifier like instance
        x_test (List[float]): A testable list (Typically from x_test)
        y_test (List[int]): A testable list (Typically from y_test)

    Returns:
        float: The prediction calculus from classifier.predict(x_test)
    """
    prediction = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(y_test, prediction)
    print(f"Precision: {precision:.2f}")
    recall = recall_score(y_test, prediction)
    print(f"Recall: {recall:.2f}")
    f1 = f1_score(y_test, prediction)
    print(f"F1 Score: {f1:.2f}")

    return prediction
