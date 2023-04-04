from Data.DatasetsStatistics.Model.Feature import CategoricalFeature, Feature, NumericFeature
import pandas as pd


class FeatureObjectCreator: 

    @staticmethod
    def get_feature_object_from_pd_dataframe(feature) -> Feature:
        if pd.api.types.is_categorical_dtype(feature):
            feature = FeatureObjectCreator.__create_categorical_feature_object(feature)   
        else:
            feature = FeatureObjectCreator.__create_numeric_feature_object(feature)
        return feature


    @staticmethod
    def __create_categorical_feature_object(feature) -> CategoricalFeature: 
        ordinal = False
        cardinality = len(feature.dtypes.categories)
        return CategoricalFeature(ordinal, cardinality)


    @staticmethod
    def __create_numeric_feature_object(feature) -> NumericFeature: 
        return NumericFeature()


 