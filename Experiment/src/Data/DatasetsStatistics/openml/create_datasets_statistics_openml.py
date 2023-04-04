import json
from pathlib import Path
from typing import Dict, List
from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
import pandas as pd 
import typer
from sklearn.datasets import fetch_openml
from utils import get_project_root

# Short Skript to fetch various datasets from open_ml to create a summary about the characteristics of the datasets 
# number of features (numeric, ordianl, nominal), cardinality

project_root = get_project_root()
datasets_root = project_root / "src" / "Data/Datasets_statistics"
path_dataset_analysis_results = datasets_root / "openml_statistics_from_jaeger_datasets"
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

datasets_statistic: DatasetStatisticsCreator = DatasetStatisticsCreator()


if __name__ == "__main__":
    for task_type in datasets.keys():
        path =  path_dataset_analysis_results / task_type
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        dataset = datasets[task_type]
        for openml_id in dataset.keys():
            try:
                data, labels = fetch_openml(data_id=openml_id, as_frame=True, return_X_y=True, cache=False)
                datasets_statistic.create_dataset_statistic_from_dataframe(openml_id, path, data)
            except:
                unsuccessfull_datasets_ids.append(openml_id)
    (path_dataset_analysis_results / "unsuccessfull_datasets.txt").write_text(str(unsuccessfull_datasets_ids))