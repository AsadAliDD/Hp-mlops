from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LogisticRegression



class Model(ABC):
    """
    Abstract class for Model
    """

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class LogisticRegressionModel(Model):
    """
    Logistic Regression Model
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=500)

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error in Training Model: {e}")
            raise e

    def predict(self, X_test):
        try:
            return self.model.predict(X_test)
        except Exception as e:
            logging.error(f"Error in Prediction: {e}")
            raise e

    def save(self, path):
        try:
            pass
        except Exception as e:
            logging.error(f"Error in Saving Model: {e}")
            raise e

    def load(self, path):
        try:
            pass
        except Exception as e:
            logging.error(f"Error in Loading Model: {e}")
            raise e