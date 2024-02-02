from zenml import pipeline
from steps.ingest_data import injest_df
from steps.clean_data import clean_df
from steps.train_model import train
from steps.evaluate_model import evaluate



@pipeline
def train_pipeline(data_path:str):
    df=injest_df(data_path)
    clean_df(df)
    train(df)
    evaluate(df)