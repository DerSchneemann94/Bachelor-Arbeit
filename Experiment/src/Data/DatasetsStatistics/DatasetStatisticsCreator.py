import json
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.DatasetsStatistics.FeatureObjectCreator import FeatureObjectCreator
from Data.DatasetsStatistics.Model.Feature import Feature
import pandas as pd
from pathlib import Path
from typing import Dict, List
from Data.DatasetsStatistics.Model.Dataset_Characteristics import CategoricalFeatureCharacteristics, DatasetCharacteristics
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher


class DatasetStatisticsCreator:


    def create_dataset_statistic_from_dataframe(self, file_name: str, path: Path, dataframe: pd.DataFrame):
        features: Dict[str, Feature] = {}
        for column in dataframe.columns: 
            features[dataframe[column].name] = FeatureObjectCreator.get_feature_object_from_pd_dataframe(dataframe[column])
        json_file = self._create_data_chacteristics_json(features)
        self.safe_results(file_name, path, json_file)


    def safe_results(self, file_name: str, path: Path, json_file):
        DatasetStatisticDaoImpl.write_json_statistic_to_file(path / (file_name + "_characteristics.json"), json_file)
        

    def _create_data_chacteristics_json(self, features: List[Feature]):
        json = {}
        for feature_key in features.keys():
            json[feature_key] = features[feature_key].get_feature_contents()
        return json

    
    def create_dataset_statistic_from_file(self, path: Path):
        dataset_statistic_json = DatasetStatisticDaoImpl.read_statistic_from_json(path)
        return dataset_statistic_json
    

    def get_datset_statistic(results_path, characteristics_path, task_type):
        datasets_statistics = {}
        #paths_to_datasets_results = PathSearcher.get_list_of_dataset_paths(results_path / task_type, "*_mean.csv")
        openml_ids = PathSearcher.get_list_of_subdirectories(results_path / task_type)
        for openml_id in openml_ids:
            path_to_dataset_results = PathSearcher. get_list_of_dataset_paths(results_path / task_type / openml_id, "*_mean.csv")
            path_to_dataset_characteristic = PathSearcher.get_list_of_dataset_paths(characteristics_path / task_type, str(openml_id) + "_characteristics.json")[0]
            result = PandasDataFrameCreator.generate_dataframe_from_paths(path_to_dataset_results)
            characteristic = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_dataset_characteristic)
            statisitic = {
                "results": result,
                "characteristic": characteristic          
            }
            datasets_statistics[openml_id]= statisitic
        return datasets_statistics
