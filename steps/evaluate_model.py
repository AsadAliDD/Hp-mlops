import logging
import pandas as pd
from zenml import step



@step
def evaluate(df: pd.DataFrame) -> None:
    """
    Evaluates the model and returns the evaluation metric

    Returns:
        float: evaluation metric
    """
    logging.info("Evaluating model")
    # # Read data from the previous step
    # data = pd.read_csv("data.csv")
    # # Train the model
    # model = Model()
    # model.train(data)
    # # Evaluate the model
    # return model.evaluate()
    pass