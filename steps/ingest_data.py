import logging
import pandas as pd 
from zenml import step



class InjestData:

    def __init__(self, path: str):
        """


        Args:
            path (str): Path to the data file
        """        
        self.path = path

    def get_data(self):
        '''Reads data from the given path and returns a pandas dataframe'''
        logging.info(f"Reading data from {self.path}")
        return pd.read_csv(self.path)



@step
def injest_data(data_path: str) -> pd.DataFrame:
    """
    Injest data from the given path and return a pandas dataframe

    Args:
        data_path (str): path to the data file 

    Returns:
        pd.DataFrame: injested dataframe 
    """    
    try:
        injest_data = InjestData(data_path)
        return injest_data.get_data()
    except Exception as e:
        logging.error(f"Failed to injest data from {data_path}")
        raise e