from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.OrdinalEncoder import OrdinalEncoderImpl
import pandas as pd
from category_encoders import LeaveOneOutEncoder
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class LeaveOneOutEncoderImpl(EncoderInterface):

    def __init__(self) -> None:
        super().__init__()


    def transform_data_human_readable(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        encoder = LeaveOneOutEncoder()
        feature_name = dataframe.columns[0]
        labels = pd.DataFrame(labels)
        if not pd.api.types.is_numeric_dtype(labels[list(labels.columns)[0]]) and len(labels[list(labels.columns)[0]].unique()) >= 2:
            labels =  OrdinalEncoderImpl().transform_data_human_readable(labels, None, None)
        try:
            encoded_dataframe = encoder.fit_transform(dataframe, labels)
            encoded_dataframe = pd.DataFrame(encoded_dataframe)
            dataframe = self.rename_column(feature_name, encoded_dataframe)
            return dataframe
        except Exception as error:
            raise error
    

    def rename_column(self, feature_name: str, encoded_dataframe: pd.DataFrame):
        current_column_name = encoded_dataframe.columns[0]
        dataframe = encoded_dataframe.rename(columns={current_column_name:feature_name})
        return dataframe
    

    def transform_data_compact(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        encoder = LeaveOneOutEncoder()
        encoder.fit(dataframe,labels)
        try:
            encoded_dataframe = encoder.transform(dataframe)
            encoded_dataframe = pd.DataFrame(encoded_dataframe)
            return encoded_dataframe
        except Exception as error:
            raise error
        
