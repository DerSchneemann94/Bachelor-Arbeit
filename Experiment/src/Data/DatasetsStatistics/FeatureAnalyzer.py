from typing import List


class FeatureAnalyzer:
    
    @staticmethod
    def get_number_of_feature_from_statistic(dataset_statistic) -> List[str]:
        #number_of_features = len(dataset_statistic.keys())
        feature_statistic = {}
        feature_composition = FeatureAnalyzer.get_composition_of_dataframe(dataset_statistic)
        for data_type in feature_composition:
            feature_statistic[data_type] = feature_composition[data_type]
        return feature_statistic


    @staticmethod
    def get_composition_of_dataframe(dataset_statistic):
        dataframe_composition = {}
        for feature_name in dataset_statistic.keys():
            feature = dataset_statistic[feature_name]
            if "subtype" not in feature:
                feature_type = feature["type"]
            else:
                feature_type = feature["subtype"]    
            if feature_type not in dataframe_composition:
                dataframe_composition[feature_type] = [feature_name]
            else:
                dataframe_composition[feature_type].append(feature_name)
        return dataframe_composition        


    @staticmethod
    def get_categorical_composition_of_dataframe(dataset_statistic):
        dataframe_composition = {}
        for feature_name in dataset_statistic.keys():
            feature = dataset_statistic[feature_name]
            if "subtype" not in feature:
                continue
            else:
                feature_type = feature["subtype"]    
            if feature_type not in dataframe_composition:
                dataframe_composition[feature_type] = [feature_name]
            else:
                dataframe_composition[feature_type].append(feature_name)
        return dataframe_composition        


    @staticmethod
    def get_cardinality_of_dataframe(dataset_statistic):
        cardinalities = {}
        for feature_name in dataset_statistic.keys():
            feature = dataset_statistic[feature_name]
            if "subtype" not in feature:
                continue
            else:
                feature_type = feature["subtype"]
            cardinality = dataset_statistic[feature_name]["cardinality"]    
            if feature_type not in cardinalities:
                cardinalities[feature_type] = [cardinality]
            else:
                cardinalities[feature_type].append(cardinality)
        return cardinalities        