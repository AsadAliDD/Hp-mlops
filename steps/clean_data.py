import logging
import pandas as pd
from zenml import step


@step
def clean_data() -> pd.DataFrame:
    """
    Cleans the data and returns a pandas dataframe

    Returns:
        pd.DataFrame: cleaned dataframe 
    """
    logging.info("Cleaning data")
    # # Read data from the previous step
    # data = pd.read_csv("data.csv")
    # # Drop rows with missing values
    # data = data.dropna()
    # return da
    pass