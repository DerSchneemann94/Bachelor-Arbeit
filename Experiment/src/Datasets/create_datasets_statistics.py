import json
from pathlib import Path
from typing import List
from Datasets.DatabaseAccessor.DatabaseAccesorDaos.OpenMl import OpenMlDatasetIdentifier
from Datasets.DatabaseAccessor.DatabaseAccesorDaos.OpenMl import OpenMlAccessor
from Datasets.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface 
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality
datasets_root = get_project_root() / "src" / "Datasets"
path_dataset_analysis_results = datasets_root / "results"
path_datasets_to_analyze = datasets_root / "DatasetsToAnalyze"




unsuccessfull_datasets_ids: List[str] = []


def analyze_data_bases():
    for openml_id in openml_ids:
        try:
            dataset_object = fetch_dataset_from_database(openml_id)
        except:
            unsuccessfull_datasets_ids.append(openml_id)



def fetch_dataset_from_database(id: str):
    return  database_accessor.get_data_set(OpenMlDatasetIdentifier(id))
 



if __name__ == "__main__":
