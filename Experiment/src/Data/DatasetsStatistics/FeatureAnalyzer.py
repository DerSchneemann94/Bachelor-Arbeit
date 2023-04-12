from typing import List


class FeatureAnalyzer:
    
    @staticmethod
    def get_number_of_feature_from_statistic(dataset_statistic) -> List[str]:
        number_of_features = len(dataset_statistic.keys())
        number_of_ordinal_features = 0
        number_of_nominal_features = 0
        for feature in dataset_statistic.keys():
            if dataset_statistic[feature].get("type") == "categorical":
                if dataset_statistic[feature].get("ordinal"):
                    number_of_ordinal_features += 1
                else:
                    number_of_nominal_features += 1
        number_of_numeric_features = number_of_features - (number_of_nominal_features + number_of_ordinal_features)        
        return number_of_nominal_features, number_of_ordinal_features, number_of_numeric_features
    


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


