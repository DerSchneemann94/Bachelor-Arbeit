from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Visualization.VisualizationDataCreator import VisualizationDataCreator
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from utils import get_project_root


root = get_project_root()
path_to_datasets_statistic = root / "src/Data/Datasets_identify/openml_statistics"
results_timestamp = "experiment_one_hot"
path_to_results = root / "results" / results_timestamp
path_to_plotting_results = root / "plot/characteristics/tables"

task_types = [
    "Binary-Classification",
    "Multiple-Classification",
    "Regression"
]

unsuccessfull_datasets = []

if __name__ == "__main__":
    datasets_statistics = {}
    datasets_feature_type_characteristics_dataframes = {}
    datasets_characteristics_dataframes = {}
    for task_type in task_types:
        datasets_statistics[task_type] = VisualizationDataCreator.get_dataset_statistic(path_to_results / task_type)
        datasets_feature_type_characteristics_dataframes[task_type] = DatasetStatisticsCreator.generate_dataframe_statistic_from_dataset_statistic(datasets_statistics[task_type])
        l
        dataframe: pd.DataFrame = datasets_feature_type_characteristics_dataframes[task_type]
        
        cells = []
        for feature_name in dataframe.columns:
            cells.append(dataframe[feature_name])
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(dataframe.columns),
            fill_color='paleturquoise', align='center'),
        cells=dict(values=cells,
            fill_color='lavender',
            align='center'
            ))
        ])
          
        path_to_plotting_results = path_to_plotting_results
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        path = path_to_plotting_results / (task_type + "_plot.html")    
        fig.write_html(path)
        
