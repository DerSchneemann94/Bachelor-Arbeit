import json
from pathlib import Path
from typing import List
import pandas as pd
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality
project_root = get_project_root()
datasets_root =  project_root / "src" / "Datasets"
path_dataset_analysis_results = datasets_root / "results"
path_datasets_to_analyze = datasets_root / "DatasetsToAnalyze"

binary_task_id_mappings = json.loads((project_root / "data/raw/binary.txt").read_text())
multi_task_id_mappings = json.loads((project_root / "data/raw/multi.txt").read_text())
regression_task_id_mappings = json.loads((project_root / "data/raw/regression.txt").read_text())

BINARY_TASK_IDS = [int(x) for x in binary_task_id_mappings.keys()]
MULTI_TASK_IDS = [int(x) for x in multi_task_id_mappings.keys()]
REGRESSION_TASK_IDS = [int(x) for x in regression_task_id_mappings.keys()]

unsuccessfull_datasets_ids: List[int] = []



def analyze_data_sets(openml_ids: List[int]):
    for openml_id in openml_ids:
        try:
            dataset_object, target_labels = fetch_dataset_from_database(openml_id)        
            determine_characteristics(dataset_object)
            print("hi")
        except:
            unsuccessfull_datasets_ids.append(openml_id)


def fetch_dataset_from_database(id: int):
    dataset, target_labels = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=False)
    return dataset, target_labels


def determine_characteristics(dataset):
    categorical_columns = [
            column for column in dataset.columns
                if pd.api.types.is_categorical_dtype(dataset[column])
    ]

    numerical_columns = [
            column for column in dataset.columns
            if pd.api.types.is_numeric_dtype(dataset[column]) and column not in categorical_columns
    ]


if __name__ == "__main__":
    analyze_data_sets(BINARY_TASK_IDS)