import logging
import pandas as pd
from zenml import step
from src.evaluation import Accuracy
from sklearn.base import ClassifierMixin

@step
def evaluate(model: ClassifierMixin,X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates the model and returns the evaluation metric

    Returns:
        float: evaluation metric
    """
    logging.info("Evaluating model")

    try:
        preds=model.predict(X_test)
        accuracy=Accuracy().evaluate(y_test,preds)
        return accuracy
    except Exception as e:
        logging.error(f"Error in Evaluating Model: {e}")
        raise e
    