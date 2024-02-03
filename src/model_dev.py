from abc import ABC, abstractmethod
import logging



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