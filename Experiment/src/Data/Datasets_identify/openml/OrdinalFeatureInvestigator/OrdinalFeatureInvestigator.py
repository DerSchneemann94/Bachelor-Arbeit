from typing import List
from Data.Datasets_identify.openml.OrdinalFeatureInvestigator.OrdinalDatasetStatisticsCreator import OrdinalDatasetStatisticsCreator
import pandas as pd
from sklearn.datasets import fetch_openml


class OrdinalFeatureInvestigator:

    def __init__(self) -> None:
        self.statistis_creator = OrdinalDatasetStatisticsCreator()
      

    def investigate_dataset(self, openml_id, dataset):
        features, labels = fetch_openml(data_id=openml_id, as_frame=True, return_X_y=True, cache=False)
        has_ordinal = self.check_if_dataset_has_ordinal_features(features)
        statistic = None
        if has_ordinal:
            statistic = self.statistis_creator.create_dataset_statistics(features, dataset)
        return statistic

    def check_if_dataset_has_ordinal_features(self, dataset) -> bool:
        categorical_columns = self.get_categorical_features(dataset)
        for categorical_feature in categorical_columns:
            feature = dataset[categorical_feature].dtype
            if feature.ordered:
                return True
        return False


    def get_categorical_features(self, dataset) -> List:
        categorical_features = [
            column for column in dataset.columns
            if pd.api.types.is_categorical_dtype(dataset[column])
        ]
        return categorical_features
    

if __name__ == "__main__":
    ordinal_investigator = OrdinalFeatureInvestigator()
    ordinal_investigator.investigate_dataset(137, None)
