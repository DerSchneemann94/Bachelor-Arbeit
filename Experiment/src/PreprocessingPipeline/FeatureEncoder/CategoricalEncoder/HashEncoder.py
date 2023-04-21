import pandas as pd
from category_encoders import HashingEncoder
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class HashEncoderImpl(EncoderInterface):

    def __init__(self) -> None:
        super().__init__()


    def transform_data_human_readable(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        encoder = HashingEncoder(return_df=True)
        encoder.fit(dataframe)
        feature_name = dataframe.columns[0]
        try:
            encoded_dataframe = encoder.transform(dataframe)
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
        encoder = HashingEncoder(return_df=True)
        encoder.fit(dataframe)
        try:
            encoded_dataframe = encoder.transform(dataframe)
            encoded_dataframe = pd.DataFrame(encoded_dataframe)
            return encoded_dataframe
        except Exception as error:
            raise error