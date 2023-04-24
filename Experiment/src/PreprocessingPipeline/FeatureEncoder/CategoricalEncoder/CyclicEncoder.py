import copy
import numpy as np
import pandas as pd
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.OrdinalEncoder import OrdinalEncoderImpl
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface

#To-DO: implement lookup table for sin and cos sin values on calculation
class CyclicEnoderImpl(EncoderInterface):

    def __init__(self, handle_unknown="ignore") -> None:
        super().__init__()
        self.handle_unknown=handle_unknown

    
    def transform_data_human_readable(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        number_of_values = len(encoding_scheme.keys())
        feature_name = dataframe.columns[0]
        pretransformed_data = OrdinalEncoderImpl().transform_data_human_readable(dataframe, labels,encoding_scheme)
        sin_column = copy.deepcopy(pretransformed_data)
        sin_column.rename(columns={feature_name:(feature_name+"_sin")}, inplace=True)
        cos_column = copy.deepcopy(pretransformed_data)
        cos_column.rename(columns={feature_name:(feature_name+"_cos")}, inplace=True)

        for row_index in range(len(pretransformed_data)):
            row = pretransformed_data.loc[row_index]
            sin_column.loc[row_index] = self.__create_cyclic_representation(row[feature_name], number_of_values, np.sin)
            cos_column.loc[row_index] = self.__create_cyclic_representation(row[feature_name], number_of_values, np.cos)
        transformed_dataframe = pd.concat([sin_column, cos_column],axis=1)
        return transformed_dataframe     


    def  __create_cyclic_representation(self, value, number_of_values, trigo):
        inner = value / number_of_values * 2 * np.pi
        result = trigo(inner)
        return result


    def transform_data_compact(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        number_of_values = len(encoding_scheme.keys())
        feature_name = dataframe.columns[0]
        pretransformed_data = OrdinalEncoderImpl().transform_data_compact(dataframe, labels, encoding_scheme)
        sin_column = copy.deepcopy(pretransformed_data)
        sin_column.rename(columns={feature_name:(feature_name+"_sin")}, inplace=True)
        cos_column = copy.deepcopy(pretransformed_data)
        cos_column.rename(columns={feature_name:(feature_name+"_cos")}, inplace=True)

        for row_index in range(len(pretransformed_data)):
            row = pretransformed_data.loc[row_index]
            sin_column.loc[row_index] = self.__create_cyclic_representation(row[feature_name], number_of_values, np.sin)
            cos_column.loc[row_index] = self.__create_cyclic_representation(row[feature_name], number_of_values, np.cos)
        transformed_dataframe = pd.concat([sin_column, cos_column],axis=1)
        return transformed_dataframe    