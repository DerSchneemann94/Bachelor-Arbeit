import json
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.DatasetsStatistics.DatasetStatisticTransformer import DatasetStatisticTransformer
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from utils import get_project_root

results_timestamp = "2023-03-31_12.51"
root = get_project_root()
path_to_datasets_statistic = root / "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_ordinal_by_hand"
path_to_results = root / "results" / results_timestamp
path_to_plotting_results = root / "plot/characteristics"


task_types = [
    "Binary-Classification",
    "Multiple-Classification",
    "Regression"
]

unsuccessfull_datasets = []


def get_datset_statistic(results_path, characteristics_path, task_type):
    datasets_statistics = {}
    #paths_to_datasets_results = PathSearcher.get_list_of_dataset_paths(results_path / task_type, "*_mean.csv")
    openml_ids = PathSearcher.get_list_of_subdirectories(results_path / task_type)
    for openml_id in openml_ids:
        path_to_dataset_results = PathSearcher. get_list_of_dataset_paths(results_path / task_type / openml_id, "*_mean.csv")
        path_to_dataset_characteristic = PathSearcher.get_list_of_dataset_paths(characteristics_path / task_type, str(openml_id) + "_characteristics.json")[0]
        result = PandasDataFrameCreator.generate_dataframe_from_paths(path_to_dataset_results)
        characteristic = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_dataset_characteristic)
        statisitic = {
            "results": result,
            "characteristic": characteristic          
        }
        datasets_statistics[openml_id]= statisitic
    return datasets_statistics


if __name__ == "__main__":
    datasets_statistics = {}
    datasets_statistics_dataframes = {}
    for task_type in task_types:
        datasets_statistics[task_type] = get_datset_statistic(path_to_results, path_to_datasets_statistic, task_type)
        datasets_statistics_dataframes[task_type] = PandasDataFrameCreator.generate_dataframe_from_dataset_statistic(datasets_statistics[task_type])

    for task_type in datasets_statistics_dataframes.keys():        
        dataframe = datasets_statistics_dataframes[task_type]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dataframe["openml_ids"],
            y=dataframe["nominal_features"],
            name='nominal_features',
            marker_color='indianred'
        )).update_traces(   
            marker={"line": {"width": 5, "color": "rgb(0,0,0)"}}
)
        fig.add_trace(go.Bar(
            x=dataframe["openml_ids"],
            y=dataframe["ordinal_features"],
            name='ordinal_features',
            marker_color='blue'
        )).update_traces(
            marker={"line": {"width": 5, "color": "rgb(0,0,0)"}}
)
        fig.add_trace(go.Bar(
            x=dataframe["openml_ids"],
            y=dataframe["numeric_features"],
            name='numeric_features',
            marker_color='green'
        )).update_traces(
            marker={"line": {"width": 5, "color": "rgb(0,0,0)"}}
)

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        
        path_to_plotting_results
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        path = path_to_plotting_results / (task_type + "_plot.html")    
        fig.write_html(path)


        
