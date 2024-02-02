import logging
import pandas as pd
from zenml import step



@step
def train_model() -> None:
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
    pass