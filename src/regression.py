import os
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_regression_model(model_path="models/regression_california_linear.pkl"):
    """
    Train a linear Regression model using California Housing dataset.
    
    Parameters:
        model_path (str): Path to save the trained pipline.

    Functionality:
        - Fetch California Housing dataset
        - Split into train/test
        - Scale features
        - Train Linear Regression
        - Print MAE, MSE, R2
        - Save trained pipeline using pickle
    """
    print("Loading California Housing dataset...")

    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
    )
    
    # Create pipline with scaling and linear regression
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression(fit_intercept=True)),
    ])

    print("Training regression model...")
    pipeline.fit(X_train, y_train)

    # Evaluate modle
    y_pred = pipeline.predict(X_test)

    print("\nRegression Metrics:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")

    # Save pipeline
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")

def use_regression_model(model_path="models/regression_california_linear.pkl"):
    """
    Load trained regression model and make predictions from user input.

    Parameters:
        model_path (str): Path to the trained pipeline.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
    except Exception as e:
        print("Error loading model: ", e)

    data = fetch_california_housing()
    feature_names = data.feature_names

    print("\nEnter values for the following features:")

    user_input = []
    for name in feature_names:
        while True:
            try:
                value = float(input(f"{name}: "))
                user_input.append(value)
                break
            except Exception as e:
                print("Invalid input. Please enter a number.")

    prediction = pipeline.predict([user_input])
    print(f"\nEstimated house value: ${prediction[0]:.2f}")