from pathlib import Path
from typing import Dict, List
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
from Experiments.ExperimentReader import ExperimentReader
from PreprocessingPipeline.FeatureEncoder.FeatureEncoder import FeatureEncoder
import pandas as pd


class CategoricalFeaturePreprocessor:

    def __init__(self, preprocessing_configuration, feature_characteristic, dataset_characteristic) -> None:
        self.__preprocessing_configuration = preprocessing_configuration
        self.__feature_characteristic = feature_characteristic
        self.__dataset_characteristic = dataset_characteristic
        self.__processor_state = self.__intialize_pipelinestate()


    def __intialize_pipelinestate(self):
        pipelinestate = []
        for data_type in self.__feature_characteristic.keys():
            pipelinestate.append({"current_state": 0, "max_state": len(self.__preprocessing_configuration[data_type])-1, "data_type":data_type})
        return pipelinestate


    def perform_preprocessing(dataframe: pd.DataFrame):
        return
    
    
    def transformData(self, dataframe: pd.DataFrame):
        transformed_data_frames_list = []
        transformation_meta_data = {}
        if self.__check_processor_state() is None:
            return None
        dataframe = self.__transform(dataframe, transformed_data_frames_list, transformation_meta_data)
        self.__adjust_processor_state()
        return dataframe


    def __transform(self, dataframe, transformed_data_frames_list, transformation_meta_data):
        dataframe_relevant_features_list = []
        for preprocessor in self.__processor_state:
            encoder_index = preprocessor["current_state"]
            data_type = preprocessor["data_type"]
            encoder_name = self.__preprocessing_configuration[data_type][encoder_index]
            dataframe_relevant_features = self.__extract_relevant_features(dataframe, self.__feature_characteristic, data_type)
            dataframe_relevant_features_list = [*dataframe_relevant_features_list, *dataframe_relevant_features]
            encoding_schemes = self.__get_relevant_encoding_schemes(dataframe_relevant_features.columns, self.__dataset_characteristic)
            dataframe_relevant_features = dataframe_relevant_features.astype(str)
            transformed_data_frames_list.append(FeatureEncoder.transform_data(encoder_name, dataframe_relevant_features, encoding_schemes))
            transformation_meta_data[data_type] = encoder_name
        transformed_dataframe = pd.concat(transformed_data_frames_list, axis=1)
        for feature_name in dataframe_relevant_features_list:
            dataframe = dataframe.drop(feature_name, axis=1)
        dataframe = pd.concat([dataframe,transformed_dataframe], axis=1)
        return dataframe


    def __check_processor_state(self):
        last_index = len(self.__processor_state)-1
        if self.__processor_state[last_index]["max_state"] < self.__processor_state[last_index]["current_state"]:
            return None
        else:
            return True


    def __adjust_processor_state(self):
        carry_bit = 1 
        last_index = len(self.__processor_state)-1
        for index in range(0, len(self.__processor_state)):
            self.__processor_state[index]["current_state"] = self.__processor_state[index]["current_state"] + carry_bit
            if (self.__processor_state[index]["max_state"] < self.__processor_state[index]["current_state"]) and index != last_index:
                self.__processor_state[index]["current_state"] = 0
                carry_bit = 1      
            else:
                carry_bit = 0
        return True


    def __extract_relevant_features(self, dataframe: pd.DataFrame, dataset_characteristics, data_type):
        features = dataset_characteristics[data_type]
        return dataframe[features]


    def __get_relevant_encoding_schemes(self, feature_names: List[str], dataset_characteristics):
        encoding_schemes = {}
        for scheme in feature_names:
            encoding_schemes[scheme] = dataset_characteristics[scheme].get("label")
        return encoding_schemes