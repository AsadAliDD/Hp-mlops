import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessing, DataSplitting
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    """
    Cleans the data and returns a pandas dataframe

    Returns:
        pd.DataFrame: cleaned dataframe 
    """
    logging.info("Cleaning data")
    try:
        preprocess_strategy=DataPreProcessing()
        data_cleaning=DataCleaning(df,preprocess_strategy)
        processed_data,label_encoder=data_cleaning.handle_data()

        split_startegy=DataSplitting()
        data_splitting=DataCleaning(processed_data,split_startegy)
        X_train,X_test,y_train,y_tes=data_splitting.handle_data()
        logging.info("Data Cleaning Completed")
        return X_train,X_test,y_train,y_tes

    except Exception as e:
        logging.error(f"Error in Data Cleaning: {e}")
        raise e
    # # Read data from the previous step
    # data = pd.read_csv("data.csv")
    # # Drop rows with missing values
    # data = data.dropna()
    # return da