import logging
from abc import ABC, abstractmethod
from typing import Union,Dict,Any,Tuple
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




class DataStrategy(ABC):
    """
    Abstract Class for Data Cleaning
    """


    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass


    
class DataPreProcessing(DataStrategy):


    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,Dict[Any, Any]]:


        try:
            data=data[[
                'Gender','Patronus','Species',
                'Blood status','Hair colour',
                'Eye colour','House']]
            data.fillna('Unknown',inplace=True)
            label_encoder={}
            for columns in data.columns:
                encoder=LabelEncoder()
                data[columns]=encoder.fit_transform(data[columns])
                label_encoder[columns]=encoder
        except Exception as e:
            logging.error(f"Error in Data Preprocessing: {e}")
            raise e
        
        return data,label_encoder
    


class DataSplitting(DataStrategy):


    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X=data.drop('House',axis=1)
            y=data['House']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test  
        except Exception as e:
            logging.error(f"Error in Data Splitting: {e}")
            raise e
        
    

class DataCleaning:
    """
    Class to handle the data cleaning process
    """

    def __init__(self, data:pd.DataFrame, strategy:DataStrategy)->None:
        self.data=data
        self.strategy=strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in Data Cleaning: {e}")
            raise e
    
    