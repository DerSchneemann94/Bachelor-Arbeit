import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class OneHotEncoderImpl(EncoderInterface):

    def __init__(self, handle_unknown="ignore") -> None:
        super().__init__()
        self.handle_unknown=handle_unknown


    def transform_data(self, dataframe: pd.DataFrame, encoding_scheme):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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
        feature_name_mapping = {}
        for feature in encoded_dataframe.columns:
            feature_name_mapping[feature] = feature_name + "_" + str(feature)
        encoded_dataframe = encoded_dataframe.rename(columns=feature_name_mapping)
        return encoded_dataframe