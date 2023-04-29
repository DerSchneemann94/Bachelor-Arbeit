from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.OrdinalEncoder import OrdinalEncoderImpl
import pandas as pd
from category_encoders import OrdinalEncoder, CatBoostEncoder
from category_encoders.wrapper import PolynomialWrapper
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class CatBoostEncoderImpl(EncoderInterface):

    def __init__(self) -> None:
        super().__init__()


    def transform_data_human_readable(self, dataframe: pd.DataFrame, labels: pd.Series, encoding_scheme):
        encoder = self.__determine_task_type_and_encoder(labels)
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
        encoder = self.__determine_task_type_and_encoder(labels)
        encoder.fit(dataframe,labels)
        try:
            encoded_dataframe = encoder.transform(dataframe)
            encoded_dataframe = pd.DataFrame(encoded_dataframe)
            return encoded_dataframe
        except Exception as error:
            raise error
        

    def __determine_task_type_and_encoder(self, labels):
        if pd.api.types.is_numeric_dtype(labels):
            return CatBoostEncoder(return_df=True)
        else:
            if len(labels.dtype.categories) > 2:
                encoder = CatBoostEncoder(return_df=True)
                return PolynomialWrapper(encoder)
            else: 
                return CatBoostEncoder(return_df=True)