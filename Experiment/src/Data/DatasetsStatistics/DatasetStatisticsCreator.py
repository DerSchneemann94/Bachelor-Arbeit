import json
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.DatasetsStatistics.FeatureObjectCreator import FeatureObjectCreator
from Data.DatasetsStatistics.Model.Feature import Feature
import pandas as pd
from pathlib import Path
from typing import Dict, List
from Data.DatasetsStatistics.Model.Dataset_Characteristics import CategoricalFeatureCharacteristics, DatasetCharacteristics


class DatasetStatisticsCreator:

    def create_dataset_statistic_from_dataframe(self, file_name: str, path: Path, dataframe: pd.DataFrame):
        features: Dict[str, Feature] = {}
        for column in dataframe.columns: 
            features[dataframe[column].name] = FeatureObjectCreator.get_feature_object_from_pd_dataframe(dataframe[column])
        json_file = self._create_data_chacteristics_json(features)
        self.safe_results(file_name, path, json_file)

    def safe_results(self, file_name: str, path: Path, json_file):
        DatasetStatisticDaoImpl.write_json_to_file(path / (file_name + "_characteristics.json"), json_file)
        

    def _create_data_chacteristics_json(self, features: List[Feature]):
        json = {}
        for feature_key in features.keys():
            json[feature_key] = features[feature_key].get_feature_contents()
        return json

    
    