import os,sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.utils import MainUtils
from src.constant import *

@dataclass
class DataTransformationConfig:
    artifact_folder = os.path.join('artifacts')
    tramsformed_train_file_path = os.path.join(artifact_folder,"train.csv")
    tramsformed_test_file_path = os.path.join(artifact_folder,"test.csv")
    transformed_obj_file_path = os.path.join(artifact_folder,"preprocessor.pkl")

class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.data_transformation_config = DataTransformationConfig()
        self.main_utils = MainUtils()
        self.feature_store_file_path = feature_store_file_path

    @staticmethod
    def get_data(feature_store_file_path):
        try:
            logging.info("Getting data")
            data = pd.read_csv(feature_store_file_path)
            logging.info("Data got successfully")
            return data
        except Exception as e:
            logging.error("Error occured while getting data")
            raise CustomException(e,sys)
        
    def get_data_transformation_object(self):
        try:
            scaler_step = ('scaler',StandardScaler())
            preprocessor = Pipeline(steps=[scaler_step])
            return preprocessor
        except Exception as e:
            logging.error("Error occured while getting data transformation object")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self):
        try:
            logging.info("Initiating data transformation")
            data = self.get_data(self.feature_store_file_path)

            X = data.drop(columns=[TARGET_COLUMN])
            y = data[TARGET_COLUMN]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
            preprocessor = self.get_data_transformation_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.transformed_obj_file_path
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
            self.main_utils.save_object(obj=preprocessor,filepath=preprocessor_path)

            train_arr = np.c_[X_train_transformed,np.array(y_train)]
            test_arr = np.c_[X_test_transformed,np.array(y_test)]

            logging.info("Saving transformed train data")

            return (train_arr,test_arr,preprocessor_path)

        except Exception as e:
            logging.error("Error occured while initiating data transformation")
            raise CustomException(e,sys)
    