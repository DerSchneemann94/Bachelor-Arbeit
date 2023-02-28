from CategoricalFeatureEncoder.CategoricalEncoderNotDefinedError import CategoricalEncoderNotDefinedError
from category_encoders import OneHotEncoder, OrdinalEncoder


categorical_encoder_dict = {
    "one_hot_encoder" : OneHotEncoder,
    "ordinal_encoder" : OrdinalEncoder
}

class Encoder:
    
    @classmethod
    def getCategoricalEncoder(categorical_encoder_name: str):
        try:   
            Encoder = categorical_encoder_dict[categorical_encoder_name]
        except:
            raise CategoricalEncoderNotDefinedError(categorical_encoder_name)
        return Encoder




