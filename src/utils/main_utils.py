import shutil
import sys
from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
import yaml

from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from yaml import safe_dump

from src.constant.training_pipeline import *
from src.exception import CustomerException
from src.logger import logging



    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomerException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomerException(e, sys)
class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomerException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)

            return schema_config

        except Exception as e:
            raise CustomerException(e, sys) from e

    def read_model_config_file(self) -> dict:
        try:
            model_config = self.read_yaml_file(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

            return model_config

        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_tuned_model(
        self,
        model_name: str,
        train_x: DataFrame,
        train_y: DataFrame,
        test_x: DataFrame,
        test_y: DataFrame,
    ) -> Tuple[float, object, str]:

        logging.info("Entered the get_tuned_model method of MainUtils class")

        try:
            model = self.get_base_model(model_name)

            model_best_params = self.get_model_params(model, train_x, train_y)

            model.set_params(**model_best_params)

            model.fit(train_x, train_y)

            preds = model.predict(test_x)

            model_score = self.get_model_score(test_y, preds)

            logging.info("Entered the get_tuned_model method of MainUtils class")

            return model_score, model, model.__class__.__name__

        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def get_model_score(test_y: DataFrame, preds: DataFrame) -> float:
        logging.info("Entered the get_model_score method of MainUtils class")

        try:
            model_score = roc_auc_score(test_y, preds)

            logging.info("Model score is {}".format(model_score))

            logging.info("Exited the get_model_score method of MainUtils class")

            return model_score

        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered the get_base_model method of MainUtils class")

        try:
            if model_name.lower().startswith("logistic") is True:
                model = linear_model.__dict__[model_name]()

            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)

                model = all_estimators().__getitem__(model_idx)[1]()

            logging.info("Exited the get_base_model method of MainUtils class")

            return model

        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_model_params(
        self, model: object, x_train: DataFrame, y_train: DataFrame
    ) -> Dict:
        logging.info("Entered the get_model_params method of MainUtils class")

        try:
            model_name = model.__class__.__name__

            model_config = self.read_model_config_file()

            model_param_grid = model_config["train_model"][model_name]

            model_grid = GridSearchCV(
                model, model_param_grid, 
                verbose = 2,
                cv = 2,
                n_jobs = -1
            )

            model_grid.fit(x_train, y_train)

            logging.info("Exited the get_model_params method of MainUtils class")

            return model_grid.best_params_

        except Exception as e:
            raise CustomerException(e, sys) from e

