import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            logging.error("Error occured while starting data ingestion")
            raise CustomException(e,sys)
    
    def start_data_transformation(self,feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path)
            train_arr,test_arr,preprocessor = data_transformation.initiate_data_transformation()
            return train_arr,test_arr,preprocessor
        except Exception as e:
            logging.error("Error occured while starting data transformation")
            raise CustomException(e,sys)
        
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer = ModelTrainer()
            model_file_path = model_trainer.initiate_model_training(X = train_arr,y = test_arr)
            return model_file_path
        except Exception as e:
            logging.error("Error occured while starting model training")
            raise CustomException(e,sys)
        
    def start_training_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr,test_arr,preprocessor = self.start_data_transformation(feature_store_file_path)
            model_file_path = self.start_model_training(train_arr,test_arr)
            return model_file_path
        except Exception as e:
            logging.error("Error occured while starting training pipeline")
            raise CustomException(e,sys)