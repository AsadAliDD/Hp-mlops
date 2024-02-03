from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import accuracy_score


class Evaluation(ABC):
    """
    Abstract class for Evaluation
    """

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class Accuracy(Evaluation):
    """
    Accuracy Evaluation
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            return accuracy_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in Accuracy Evaluation: {e}")
            raise e