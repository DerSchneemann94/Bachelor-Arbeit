import json
from pathlib import Path
from typing import Dict, List
from Data.Datasets_identify.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface
from Data.Datasets_identify.Model.Dataset_Characteristics import CategoricalFeatureCharacteristics, DatasetCharacteristics
import pandas as pd 
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality

project_root = get_project_root()
datasets_root = project_root / "src" / "Data/Datasets_identify"
path_dataset_analysis_results = datasets_root / "openml_statistics"
path_datasets_to_analyze = datasets_root / "DatasetsToAnalyze"

binary_task_id_mappings = json.loads((project_root / "data/raw/binary.txt").read_text())
multi_task_id_mappings = json.loads((project_root / "data/raw/multi.txt").read_text())
regression_task_id_mappings = json.loads((project_root / "data/raw/regression.txt").read_text())

datasets = {
    "Binary-Classification": binary_task_id_mappings,
    "Multiple-Classification": multi_task_id_mappings,
    "Regression": regression_task_id_mappings
}
unsuccessfull_datasets_ids: List[str] = []


def analyze_data_sets(openml_ids) -> Dict[str, DatasetCharacteristics]:
    dataset_categorical_features_statistic = {}
    for openml_id in openml_ids:
        try:
            dataset = fetch_dataset_from_database(openml_id)
            dataset_categorical_characteristics = determine_categorical_feature_characteristics(dataset)
            if dataset_categorical_characteristics:
                number_of_features = dataset.columns.size
                number_of_instances = dataset.index.size
                dataset_categorical_features_statistic[openml_id] =  DatasetCharacteristics(number_of_instances, number_of_features, dataset_categorical_characteristics) 
        except Exception as error:
            unsuccessfull_datasets_ids.append(openml_id)
            #print("Dataset_Id: ", openml_id)
            continue
    return dataset_categorical_features_statistic    


def determine_categorical_feature_characteristics(dataset) -> List[CategoricalFeatureCharacteristics]:
    categorical_columns = [
        column for column in dataset.columns
        if pd.api.types.is_categorical_dtype(dataset[column])
    ]
    categorical_features_characteristics: List[CategoricalFeatureCharacteristics] = []
    for categorical_feature in categorical_columns:
        feature = dataset[categorical_feature].dtype
        ordered = feature.ordered
        cardinality = len(feature.categories)
        characteristics = CategoricalFeatureCharacteristics(ordered, cardinality, categorical_feature)
        categorical_features_characteristics.append(characteristics)
    return categorical_features_characteristics


def fetch_dataset_from_database(id: str) -> pd.DataFrame:
    dataset, target_labels = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=False)
    return dataset


def safe_results(path: Path, datasets_statistic: Dict[str, DatasetCharacteristics]):
    for openml_id in datasets_statistic.keys():
        with open(path / (openml_id + "_characteristics.txt"), "a") as file:
            dataset_statistic = datasets_statistic[openml_id]   
            number_of_features = dataset_statistic.features
            number_of_instances = dataset_statistic.instances
            dataset_categorical_statistic = dataset_statistic.categorical_feature_characteristics
            ordinal_cardinality = []
            nominal_cardinality = []
            for feature_characteristic in dataset_categorical_statistic:
                file.write(feature_characteristic.name + ":\t\tcardinality-" + str(feature_characteristic.cardinality) + ",\t\tordinal-" + str(feature_characteristic.ordinal) + "\n")
                if feature_characteristic.ordinal:
                    ordinal_cardinality.append(str(feature_characteristic.cardinality))
                else:
                    nominal_cardinality.append(str(feature_characteristic.cardinality))
            file.write("features: " + str(number_of_features) + "\n" )
            file.write("instances: " + str(number_of_instances) + "\n" )
            file.write("ordinal_cardinality: " + str(ordinal_cardinality) + "\n" )
            file.write("nominal_cardinality: " + str(nominal_cardinality) + "\n")


def safe_corrupted_datasets(path: Path):
    with open(path / "corrupted_datasets.txt", "w") as file:
        file.write(str(unsuccessfull_datasets_ids))


if __name__ == "__main__":
   for task_type in datasets.keys():
        path =  path_dataset_analysis_results / task_type
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        datasets_statistic = analyze_data_sets(datasets[task_type])
        safe_results(path, datasets_statistic)
        path = path_dataset_analysis_results 
        safe_corrupted_datasets(path)
   