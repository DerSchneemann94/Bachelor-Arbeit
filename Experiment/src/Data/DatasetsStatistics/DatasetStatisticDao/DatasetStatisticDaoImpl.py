from abc import ABC, abstractmethod
import json
from pathlib import Path

import pandas as pd

from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDao import DatasetStatisticDao


class DatasetStatisticDaoImpl(DatasetStatisticDao):

    @staticmethod
    def write_json_statistic_to_file(path, serializable_file):
        try:
            with open(path, "w") as file:
                json.dump(serializable_file, file, indent=4, sort_keys=True)
        except Exception as error: 
            print("Error occured while writing to file " + str(path))
            raise error

    
    @staticmethod
    def write_dataframe_statistic_to_file(path: Path, dataframe: pd.DataFrame):
        try:
            dataframe.to_csv(path)
        except Exception as error:
            print("Failed to write dataframe to:  " + str(path))
            raise error


    @staticmethod
    def read_statistic_from_json(path):
        dict = None 
        try: 
            with open(path, "r") as file:
                dict = json.load(file)
        except Exception as error: 
            print("Error occured while reading from file " + str(path))
            raise error 
        return dict
    
