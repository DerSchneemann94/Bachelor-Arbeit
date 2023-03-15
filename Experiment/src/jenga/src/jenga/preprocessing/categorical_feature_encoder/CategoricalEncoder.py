from category_encoders import OneHotEncoder, OrdinalEncoder
from jenga.preprocessing.categorical_feature_encoder.CategoricalEncoderNotDefinedError import CategoricalEncoderNotDefinedError
from jenga.preprocessing.categorical_feature_encoder.NumericalEncoderNotDefinedError import NumericalEncoderNotDefinedError
from sklearn.preprocessing import StandardScaler




categorical_encoder_dict = {
    "one_hot_encode" : OneHotEncoder,
    "ordinal_encode" : OrdinalEncoder
}

numerical_encoder_dict = {
    "scaling" : StandardScaler
}

class Encoder:
    @staticmethod
    def getCategoricalEncoder(categorical_encoder_name: str):
        try:   
            encoder = categorical_encoder_dict[categorical_encoder_name]
        except:
            raise CategoricalEncoderNotDefinedError(categorical_encoder_name)
        return encoder


    @staticmethod
    def getNumericalEncoder(numerical_encoder_name: str):
        try:   
            encoder = numerical_encoder_dict[numerical_encoder_name]
        except:
            raise NumericalEncoderNotDefinedError(numerical_encoder_name)
        return encoder

