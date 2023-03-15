import json
from pathlib import Path
from typing import List
from Data.Datasets_identify.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface
from Data.Datasets_identify.Model.CategoricalFeatureCharacteristics import CategorialFeatureCharacteristics
import pandas as pd 
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality

project_root = get_project_root()
datasets_root = project_root / "src" / "Datasets"
path_dataset_analysis_results = datasets_root / "results"
path_datasets_to_analyze = datasets_root / "DatasetsToAnalyze"

binary_task_id_mappings = json.loads((project_root / "data/raw/binary.txt").read_text())
multi_task_id_mappings = json.loads((project_root / "data/raw/multi.txt").read_text())
regression_task_id_mappings = json.loads((project_root / "data/raw/regression.txt").read_text())

unsuccessfull_datasets_ids: List[str] = []


def analyze_data_sets(openml_ids):
    dataset_statistics_collection = {}
    for openml_id in openml_ids:
        try:
            dataset = fetch_dataset_from_database(openml_id)
            dataset_categorical_characteristics = determine_categorical_feature_characteristics(dataset)
            if not dataset_categorical_characteristics:
                dataset_statistics_collection[openml_id] = dataset_categorical_characteristics
        except Exception as error:
            unsuccessfull_datasets_ids.append(openml_id)
            #print("Dataset_Id: ", openml_id)
            raise error

    print("hi")

def determine_categorical_feature_characteristics(dataset):
    categorical_columns = [
        column for column in dataset.columns
        if pd.api.types.is_categorical_dtype(dataset[column])
    ]
    categorical_features_characteristics: List[CategorialFeatureCharacteristics] = []
    for categorical_feature in categorical_columns:
        ordered = False
        cardinality = []
        categorical_data = dataset[categorical_feature].dtype()
        if categorical_data.ordered():
            ordered = True
        characteristics = CategorialFeatureCharacteristics(ordered, cardinality, categorical_feature)
        categorical_features_characteristics.append(characteristics)
    return categorical_features_characteristics

def fetch_dataset_from_database(id: str):
    dataset, target_labels = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=False)
    return dataset


if __name__ == "__main__":
    analyze_data_sets(binary_task_id_mappings)
   