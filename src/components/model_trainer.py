import os,sys
from dataclasses import dataclass
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
from src.constant import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join('artifacts')
    model_file_path = os.path.join(artifact_folder,"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.main_utils = MainUtils()
        self.model = SVC(C=1.0,kernel='rbf',gamma='auto')

    def initiate_model_training(self,X,y):
        try:
            X_train, y_train, X_test, y_test = (
                X[:,:-1],X[:,-1],
                y[:,:-1],y[:,-1]
            )
            logging.info("Initiating model training")
            self.model.fit(X_train,y_train)
            y_pred = self.model.predict(X_test)
            logging.info("Model trained successfully")
            accuracy = accuracy_score(y_test,y_pred)
            logging.info("Accuracy: {}".format(accuracy))
            os.makedirs(os.path.dirname(self.model_trainer_config.model_file_path),exist_ok=True)
            self.main_utils.save_object(obj=self.model,filepath=self.model_trainer_config.model_file_path)
            logging.info("Model saved successfully")
            return self.model_trainer_config.model_file_path
        except Exception as e:
            logging.error("Error occured while training model")
            raise CustomException(e,sys)