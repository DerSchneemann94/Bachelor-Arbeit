import pandas as pd
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.CyclicEncoder import CyclicEnoderImpl
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.OneHotEncoder import OneHotEncoderImpl
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.OrdinalEncoder import OrdinalEncoderImpl
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.HashEncoder import HashEncoderImpl
from PreprocessingPipeline.FeatureEncoder.CategoricalEncoder.GlmmEncoder import GlmmEncoderImpl
from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface
from PreprocessingPipeline.FeatureEncoder.Exceptions.CategoricalEncoderNotDefinedError import CategoricalEncoderNotDefinedError
from PreprocessingPipeline.FeatureEncoder.Exceptions.NumericalEncoderNotDefinedError import NumericalEncoderNotDefinedError
from sklearn.preprocessing import StandardScaler


categorical_encoder_dict = {
    "ordinal" : OrdinalEncoderImpl(handle_unknown="use_encoded_value", unknown_value=-1),
    "one_hot" : OneHotEncoderImpl(handle_unknown="ignore"),
    "cyclic" : CyclicEnoderImpl(handle_unknown="ignore"),
    "hashing": HashEncoderImpl(),
    "glmm": GlmmEncoderImpl(),
}

numerical_encoder_dict = {
    "scaling" : StandardScaler
}

class FeatureEncoder:
    @staticmethod
    def get_categorical_encoder(categorical_encoder_name: str) -> EncoderInterface:
        try:   
            encoder = categorical_encoder_dict[categorical_encoder_name]
        except:
            raise CategoricalEncoderNotDefinedError(categorical_encoder_name)
        return encoder


    @staticmethod
    def get_numerical_encoder(numerical_encoder_name: str) -> EncoderInterface:
        try:   
            encoder = numerical_encoder_dict[numerical_encoder_name]
        except:
            raise NumericalEncoderNotDefinedError(numerical_encoder_name)
        return encoder  


    @staticmethod
    def transform_data(encoder_name: str, dataframe:pd.DataFrame, labels:pd.Series, encoding_schemes, human_readable) -> pd.DataFrame:
        try: 
            encoder = FeatureEncoder.get_categorical_encoder(encoder_name)
        except:
            encoder = FeatureEncoder.get_numerical_encoder(encoder_name)
        
        list_of_transformed_features = []
        for feature_name in dataframe.columns:
            encoding_scheme = encoding_schemes[feature_name]
            feature_vector = pd.DataFrame(dataframe[feature_name])
            feature_vector = feature_vector.astype(str)
            if human_readable:
                data = encoder.transform_data_human_readable(feature_vector, labels, encoding_scheme)
            else:
                data = encoder.transform_data_compact(feature_vector, labels, encoding_scheme)
            transformed_dataframe = pd.DataFrame(data)
            list_of_transformed_features.append(transformed_dataframe)
        transformed_dataframe = pd.concat(list_of_transformed_features, axis=1)
        return transformed_dataframe


    
