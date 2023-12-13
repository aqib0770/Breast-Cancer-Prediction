import sys, os
from typing import Dict,Tuple
import pandas as pd
import pickle
import yaml
from src.constant import *
from src.logger import logging
from src.exception import CustomException

class MainUtils:
    def __init__(self)->None:
        pass
    
        
    def read_schema_config_file(self)->dict:
        try:
            schema_config=self.read_yaml(os.path.join("config","schema.yaml"))
            return schema_config
        except Exception as e:
            logging.error("Error occured while reading schema config file")
            raise CustomException(e,sys)
        
    @staticmethod
    def save_object(filepath:str,obj:object):
        logging.info("Saving object")
        try:
            with open(filepath,"wb") as file:
                pickle.dump(obj,file)
            logging.info("Object saved successfully")
        except Exception as e:
            logging.error("Error occured while saving object")
            raise CustomException(e,sys)
        
    @staticmethod
    def load_object(filepath):
        logging.info("Loading object")
        try:
            with open(filepath,"rb") as file:
                obj=pickle.load(file)
            logging.info("Object loaded successfully")
            return obj
        except Exception as e:
            logging.error("Error occured while loading object")
            raise CustomException(e,sys)
        
    @staticmethod
    def load_object(file_path:str)->object:
        logging.info("Loading object")
        try:
            with open(file_path,"rb") as file:
                obj=pickle.load(file)
            logging.info("Object loaded successfully")
            return obj
        except Exception as e:
            logging.error("Error occured while loading object")
            raise CustomException(e,sys)