from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import MainUtils
from src.constant import *
import sys, os
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import numpy as np
import pandas as pd

@dataclass
class DataIngestionConfig:
    artifact_folder = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.main_utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name,db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]
            data = pd.DataFrame(list(collection.find()))

            if "_id" in data.columns:
                data.drop(columns=["_id"],inplace=True)
            
            return data

        except Exception as e:
            logging.error("Error occured while exporting collection as dataframe")
            raise CustomException(e,sys)
        
    def export_data_into_feature(self)->pd.DataFrame:
        try:
            logging.info("Exporting data into feature")
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path,exist_ok=True)

            data = self.export_collection_as_dataframe(COLLECTION_NAME,DATABASE_NAME)
            logging.info("Data exported successfully")

            feature_store_file_path = os.path.join(raw_file_path,"breast_cancer_data.csv")
            logging.info("Saving data into feature store: {0}".format(feature_store_file_path))
            data.to_csv(feature_store_file_path,index=False)

            logging.info("Data saved successfully")
            return feature_store_file_path

        except Exception as e:
            logging.error("Error occured while exporting data into feature")
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            feature_store_file_path = self.export_data_into_feature()
            logging.info("Data ingestion initiated successfully")
            return feature_store_file_path
        except Exception as e:
            logging.error("Error occured while initiating data ingestion")
            raise CustomException(e,sys)