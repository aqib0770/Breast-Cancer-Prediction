import json
import pandas as pd
from pymongo import MongoClient
from src.constant import *

client = MongoClient(MONGO_DB_URL)
db = DATABASE_NAME
collection = COLLECTION_NAME

data = pd.read_csv('notebooks/breast_cancer.csv')

json_data=  list(json.loads(data.T.to_json()).values())

client[db][collection].insert_many(json_data)