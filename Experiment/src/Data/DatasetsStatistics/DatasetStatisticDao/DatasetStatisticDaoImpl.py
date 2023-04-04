from abc import ABC, abstractmethod
import json

from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDao import DatasetStatisticDao


class DatasetStatisticDaoImpl(DatasetStatisticDao):

    @staticmethod
    def write_json_to_file(path, serializable_file):
        try:
            with open(path, "w") as file:
                json.dump(serializable_file, file, indent=4, sort_keys=True)
        except Exception as error: 
            print("Error occured while writing to file " + path)
            raise error


    @staticmethod
    def read_from_json(path):
        dict = None 
        try: 
            with open(path, "r") as file:
                dict = json.load(file)
        except Exception as error: 
            print("Error occured while reading from file " + path)
            raise error 
        return dict
    
