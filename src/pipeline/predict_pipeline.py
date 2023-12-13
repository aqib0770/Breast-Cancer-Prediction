import os,sys
from dataclasses import dataclass
import pandas as pd
from flask import request
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils import MainUtils

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname:str = "predictions"
    prediction_output_filename:str = "predictions.csv"
    model_file_path:str = os.path.join('artifacts','model.pkl')
    preprocessor_file_path:str = os.path.join('artifacts','preprocessor.pkl')
    prediction_output_path:str = os.path.join(prediction_output_dirname,prediction_output_filename)


class PredictionPipeline:
    def __init__(self,request:request):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.main_utils = MainUtils()
        self.request = request

    def save_prediction(self):
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir,exist_ok=True)
            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir,input_csv_file.filename)
            input_csv_file.save(pred_file_path)
            return pred_file_path
        except Exception as e:
            logging.error("Error occured while saving prediction")
            raise CustomException(e,sys)
        
    def predict(self,features):
        try:
            model_path = self.prediction_pipeline_config.model_file_path
            preprocessor_path = self.prediction_pipeline_config.preprocessor_file_path
            
            model = self.main_utils.load_object(model_path)
            preprocessor = self.main_utils.load_object(preprocessor_path)

            transformed_features = preprocessor.transform(features)
            predictions = model.predict(transformed_features)

            return predictions
        except Exception as e:
            logging.error("Error occured while predicting")
            raise CustomException(e,sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
        try:
            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

            input_dataframe = input_dataframe.drop(columns = "Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]

            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname,exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_output_path)
            logging.info("Predictions completed")
        except Exception as e:
            logging.error("Error occured while getting predicted dataframe")
            raise CustomException(e,sys)
        
    def run_pipeline(self):
        try:
            logging.info("Running prediction pipeline")
            input_dataframe_path = self.save_prediction()
            self.get_predicted_dataframe(input_dataframe_path)
            logging.info("Prediction pipeline run successfully")
            return self.prediction_pipeline_config
        except Exception as e:
            logging.error("Error occured while running prediction pipeline")
            raise CustomException(e,sys)