import logging
import pandas as pd
from zenml import step
from src.model_dev import LogisticRegressionModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

@step
def train(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, config: ModelNameConfig
) -> ClassifierMixin:
    """
    Trains the model and saves it to the model directory
    """
    logging.info("Training model")

    
    # # Read data from the previous step
    # data = pd.read_csv("data.csv")
    # # Train the model
    # model = Model()
    # model.train(data)
    # # Save the model
    # model.save("model")
    try:
        if config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()
            return model.train(X_train, y_train)
        else:
            raise ValueError("Model name not found")
    except Exception as e:
        logging.error(f"Error in Training Model: {e}")
        raise e
