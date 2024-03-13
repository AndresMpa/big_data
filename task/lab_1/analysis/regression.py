import pandas as pd
from util.env_vars import config
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def linear_regression(data: pd.DataFrame, target: pd.Series):
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


def test_regression(regression, x_test, y_test) -> float:
    prediction = regression.predict(x_test)

    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_test, prediction)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_test, prediction)
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Calculate R-squared (coefficient of determination)
    r2 = r2_score(y_test, prediction)
    print(f"R-squared (R2): {r2:.2f}")

    return prediction


def test_classification(classifier, x_test, y_test) -> float:
    prediction = classifier.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate precision
    precision = precision_score(y_test, prediction)
    print(f"Precision: {precision:.2f}")

    # Calculate recall
    recall = recall_score(y_test, prediction)
    print(f"Recall: {recall:.2f}")

    # Calculate F1 score
    f1 = f1_score(y_test, prediction)
    print(f"F1 Score: {f1:.2f}")

    return prediction
