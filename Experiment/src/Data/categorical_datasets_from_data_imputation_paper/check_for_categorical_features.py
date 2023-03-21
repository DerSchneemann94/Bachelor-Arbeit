import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import List
from sklearn.datasets import fetch_openml
from utils import get_project_root

timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M")
project_root = get_project_root()
datasets_root =  project_root / "src" / "Data"
path_datasets_openml = datasets_root / "categorical_datasets_from_data_imputation_paper" / timestamp

binary_task_id_mappings = json.loads((project_root / "data/raw/binary.txt").read_text())
multi_task_id_mappings = json.loads((project_root / "data/raw/multi.txt").read_text())
regression_task_id_mappings = json.loads((project_root / "data/raw/regression.txt").read_text())

BINARY_TASK_IDS = [int(x) for x in binary_task_id_mappings.keys()]
MULTI_TASK_IDS = [int(x) for x in multi_task_id_mappings.keys()]
REGRESSION_TASK_IDS = [int(x) for x in regression_task_id_mappings.keys()]

failed_datasets_list: List[int] = []


def analyze_datasets(dataset_ids, task_type):
    for dataset_id in dataset_ids:
        try:
            dataset, labels = fetch_openml(data_id=dataset_id, as_frame=True, return_X_y=True, cache=False)
            categorical_features = []
            for feature in dataset:
                if pd.api.types.is_categorical_dtype(dataset[feature]):
                    categorical_features.append([feature, get_cardinality_of_feature(dataset[feature])])
            if categorical_features:
                safe_results(dataset_id, task_type, dataset, labels, categorical_features)
        except Exception as error:        
            failed_datasets_list.append(dataset_id)
            #raise error


def safe_results(dataset_id, task_type, dataset, labels, categorical_features):
    path_to_dataset = path_datasets_openml / task_type / str(dataset_id)
    path_to_dataset.mkdir(parents=True, exist_ok=True)
    safe_dataset(dataset, labels, path_to_dataset, dataset_id)
    safe_data_set_statistics(categorical_features, path_to_dataset)


def get_cardinality_of_feature(feature: pd.Series) -> int:
    return feature.nunique()


def safe_data_set_statistics(categorical_features, path_to_dataset):
    with open(path_to_dataset / "statistics.txt", "a") as file:
        file.write("Cardinality of categorical features:\n")
        for feature in categorical_features:
            file.write(feature[0] + ": " + str(feature[1]) + "\n")

def safe_dataset(dataset: pd.DataFrame, labels: pd.Series, path_to_dataset: Path, dataset_id: int):
    dataset[labels.name] = labels
    dataset.to_csv(path_to_dataset / (str(dataset_id) + ".csv"), index=False)


analyze_datasets(BINARY_TASK_IDS, "Binary-Classification")
analyze_datasets(MULTI_TASK_IDS, "Multiple-Classification")
analyze_datasets(REGRESSION_TASK_IDS, "Regression")


with open(path_datasets_openml / "failed_datasets.txt", "w") as file:
    write_to_file =  str(failed_datasets_list)
    file.write(write_to_file)