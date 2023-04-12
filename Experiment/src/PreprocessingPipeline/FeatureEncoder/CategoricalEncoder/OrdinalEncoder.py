import pandas as pd
from category_encoders import OrdinalEncoder
#from sklearn.preprocessing import OrdinalEncoder
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class OrdinalEncoderImpl(EncoderInterface):

    def __init__(self,handle_unknown="use_encoded_value", unknown_value=-1) -> None:
        super().__init__()
        self.handle_umknown=handle_unknown
        self.unknown_value=unknown_value


    def transform_data(self, dataframe: pd.DataFrame, encoding_scheme):
        encoder = OrdinalEncoder(cols = dataframe.columns[0],handle_unknown="error", mapping=[{"col": dataframe.columns[0],"mapping":encoding_scheme}])
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