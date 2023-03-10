import json
from pathlib import Path
from typing import List
from Datasets.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface 
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality

project_root = get_project_root()
datasets_root = project_root / "src" / "Datasets"
path_dataset_analysis_results = datasets_root / "results"
path_datasets_to_analyze = datasets_root / "DatasetsToAnalyze"

binary_task_id_mappings = json.loads(project_root / "data/raw/binary.txt").read_text()
multi_task_id_mappings = json.loads(project_root / "data/raw/multi.txt").read_text()
regression_task_id_mappings = json.loads(project_root / "data/raw/regression.txt").read_text()

unsuccessfull_datasets_ids: List[str] = []

def get_openml_ids(dataset_list):
    pass
    



def analyze_data_bases(openml_ids):
    for openml_id in openml_ids:
        try:
            dataset_object = fetch_dataset_from_database(openml_id)
        except:
            unsuccessfull_datasets_ids.append(openml_id)



def fetch_dataset_from_database(id: str):
    return  database_accessor.get_data_set(OpenMlDatasetIdentifier(id))
 



if __name__ == "__main__":
    pass
