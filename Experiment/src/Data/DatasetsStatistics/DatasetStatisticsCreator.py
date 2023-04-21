import json
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
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


    @staticmethod
    def create_dataset_statistic_from_dataframe_openml( dataframe: pd.DataFrame):
        features: Dict[str, Feature] = {}
        for column in dataframe.columns: 
            features[dataframe[column].name] = FeatureObjectCreator.get_feature_object_from_pd_dataframe(dataframe[column])
        json_file = DatasetStatisticsCreator.__create_data_chacteristics_json(features)
        return json_file
           

    @staticmethod
    def __create_data_chacteristics_json(features: List[Feature]):
        json = {}
        for feature_key in features.keys():
            json[feature_key] = features[feature_key].get_feature_contents()
        return json


    @staticmethod
    def create_dataset_statistic_from_file(path: Path):
        dataset_statistic_json = DatasetStatisticDaoImpl.read_statistic_from_json(path)
        return dataset_statistic_json

    
    @staticmethod
    def generate_dataframe_statistic_from_dataset_statistic(datasets_statistic):
        openml_ids = []
        features_types_sorted = []
        datasets_instances = []
        dataframe_initialzier = {}    
        for openml_id in datasets_statistic.keys():
            number_of_entries = len(openml_ids)
            dataset_statistic = datasets_statistic[openml_id]
            feature_characteristic = dataset_statistic["feature_characteristic"]
            try:
                features_types_sorted = FeatureAnalyzer.get_number_of_feature_from_statistic(feature_characteristic) 
                dataframe_initialzier = DatasetStatisticsCreator.__update_statistic_dataframe(features_types_sorted, dataframe_initialzier, number_of_entries)
            except Exception as error:
                continue
            openml_ids.append(openml_id)
            instances = dataset_statistic["dataset_characteristic"]["instances"]
            datasets_instances.append(instances)
        dataframe_initialzier = {**{"openml_ids":openml_ids}, **{"instances": datasets_instances}, **dataframe_initialzier}     
        dataframe_performance = pd.DataFrame(dataframe_initialzier)
        return dataframe_performance


    @staticmethod
    def __update_statistic_dataframe(feature_types, dataframe_initializer_json, number_of_entries):
        feature_types_keys: List = feature_types.keys()
        dataframe_initializer_json_keys: List = list(dataframe_initializer_json.keys())

        for data_type in feature_types_keys:
            number_of_features = len(feature_types[data_type])  
            if data_type in dataframe_initializer_json.keys():
                dataframe_initializer_json[data_type].append(number_of_features)
                dataframe_initializer_json_keys.remove(data_type)
            else:
                dataframe_initializer_json[data_type] = [0] * number_of_entries    
                dataframe_initializer_json[data_type].append(number_of_features)
                    
        for data_type in dataframe_initializer_json_keys:
            dataframe_initializer_json[data_type].append(0)
        return dataframe_initializer_json

