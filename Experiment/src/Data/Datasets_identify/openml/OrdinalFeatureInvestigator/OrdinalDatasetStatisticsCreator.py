import pandas as pd


class OrdinalDatasetStatisticsCreator:

    def __init__(self) -> None:
        pass

    
    def create_dataset_statistics(self, features, dataset_metadata) -> None:
        number_of_features = dataset_metadata.NumberOfFeatures
        number_of_instances = dataset_metadata.NumberOfInstances
        number_of_missing_values = dataset_metadata.NumberOfMissingValues
        nominal_cardinality, ordinal_cardinality = self.determine_cardinality(features)
        statistics = {
            "number_of_features":number_of_features,
            "number_of_instances":number_of_instances,
            "number_of_missing_values":number_of_missing_values,
            "nominal_cardinality":str(nominal_cardinality),
            "ordinal_cardinality":str(ordinal_cardinality),
        }
        return statistics

    def determine_cardinality(self, dataframe):
        feature_names = dataframe.columns
        nominal_cardinality, ordinal_cardinality = []
        for feature_name in feature_names:
            feature = dataframe[feature_name]
            if pd.api.types.is_categorical_dtype(feature):
                if feature.ordered:
                    ordinal_cardinality.append(len(feature.categories))
                else:
                    nominal_cardinality.append(len(feature.categories))
        return nominal_cardinality, ordinal_cardinality