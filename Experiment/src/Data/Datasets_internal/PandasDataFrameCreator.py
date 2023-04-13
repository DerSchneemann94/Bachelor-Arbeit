from typing import Dict, List
import pandas as pd

from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
from pathlib import Path

class PandasDataFrameCreator:

    @staticmethod
    def generate_dataframe_from_path(path:Path):
        try:
            dataframe = pd.read_csv(path)
            return dataframe
        except Exception as error:
            raise error
        


    @staticmethod
    def generate_dataframe_from_paths(paths: Path):
        dataframes = []
        for path in paths:
            dataframe = pd.read_csv(path, skipinitialspace=True, index_col=False)
            dataframes.append(dataframe)
        results_dataframe: pd.DataFrame = pd.concat(dataframes, axis=1)
        results_dataframe = results_dataframe.loc[:, ~results_dataframe.columns.duplicated()].copy()
        new_names_dict = {}
        for column in results_dataframe.columns:
            if "encoder:" in column:
                encoder_long = column.split("encoder: ")[1]
                encoder = encoder_long.split("_encode")[0]
                new_names_dict[column] = encoder
        results_dataframe = results_dataframe.rename(columns=new_names_dict)        
        return results_dataframe


    @staticmethod
    def split_dataframe_into_data_and_labels(dataframe: pd.DataFrame, target: str):
        label = pd.Series(dataframe[target])
        dataframe = dataframe.drop(target, axis=1)
        return dataframe, label


    @staticmethod 
    def generate_dataframe_from_json(json):
        pass


    @staticmethod
    def generate_dataframe_from_dataset_statistic(datasets_statistic):
        openml_ids = []
        features_charateristics = []
        
        dataframe_initialzier = {}    
        for openml_id in datasets_statistic.keys():
            number_of_entries = len(openml_ids)
            dataset_statistic = datasets_statistic[openml_id]
            try:
                features_charateristics = FeatureAnalyzer.get_number_of_feature_from_statistic(dataset_statistic["dataset_characteristic"]) 
                dataframe_initialzier = PandasDataFrameCreator.__update_statistic_dataframe(features_charateristics, dataframe_initialzier, number_of_entries)
            except Exception as error:
                raise error
            openml_ids.append(openml_id)
        dataframe_initialzier["openml_ids"] = openml_ids     
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



    @staticmethod
    def generate_dataframe_from_dataset_results(datasets_results, metric):
        openml_ids = []
        results_encoders = {}
        example_id = list(datasets_results.keys())[0]
        dataset_for_initialization = datasets_results[example_id]["results"]
        dataset_for_initialization = dataset_for_initialization.drop(dataset_for_initialization.columns[0], axis=1)
        for encoder in dataset_for_initialization.columns:
            results_encoders[encoder] = []
        for openml_id in datasets_results.keys():
            try:
                results:pd.DataFrame = datasets_results[openml_id]["results"]
                encoder_performance = results.drop(results.columns[0], axis=1)
                results = results.set_index(results.columns[0])
                for encoder in encoder_performance.columns:
                    value = results.at[metric, encoder]
                    results_encoders[encoder].append(value)
                openml_ids.append(openml_id)
                results_encoders["openml_ids"] = openml_ids
                dataframe = pd.DataFrame(results_encoders)    
            except Exception as error:
                continue
        return dataframe

    # @staticmethod
    # def generate_results_dataframe(results: Dict):
    #     dataframe_list = []
    #     for openml_id in results.keys():
    #         dataframe_list.append(pd.DataFrame(results[openml_id]["results"]))
    #     dataframe = pd.concat(dataframe_list, axis=1)    
    #     dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()].copy()
    #     return dataframe