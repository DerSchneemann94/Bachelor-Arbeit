from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
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


def add_cardinality_to_dataframe(dataframe:pd.DataFrame, datasets_statistics):
    data_frame_features = dataframe.drop(columns=["openml_ids", "instances", "numeric"], axis=1)
    cardinalities = {}
    for data_type in data_frame_features.columns:
        cardinalities[data_type] = []
    for openml_id in  dataframe["openml_ids"]:
        dataset_cardinality = FeatureAnalyzer.get_cardinality_of_dataframe(datasets_statistics[openml_id]["feature_characteristic"])
        for data_type in data_frame_features:
            if data_type in dataset_cardinality.keys():
                cardinalities[data_type].append(str(dataset_cardinality[data_type]))
            else:
                cardinalities[data_type].append("-")
    return cardinalities


if __name__ == "__main__":
    datasets_statistics = {}
    datasets_feature_type_characteristics_dataframes = {}
    datasets_characteristics_dataframes = {}
    for task_type in task_types:
        datasets_statistics = VisualizationDataCreator.get_dataset_statistic(path_to_results / task_type)
        datasets_feature_type_characteristics_dataframes = DatasetStatisticsCreator.generate_dataframe_statistic_from_dataset_statistic(datasets_statistics)
        dataframe: pd.DataFrame = datasets_feature_type_characteristics_dataframes
        cardinalities = add_cardinality_to_dataframe(dataframe, datasets_statistics)
        carindialites_dataframe = pd.DataFrame(cardinalities)
        mapping = {}
        for data_type in carindialites_dataframe.columns:
            mapping[data_type] = data_type + "_cardinality"
        carindialites_dataframe.rename(columns=mapping, inplace=True)
        dataframe = pd.concat([dataframe, carindialites_dataframe], axis=1)
        cells = []
        for feature_name in dataframe.columns:
            cells.append(dataframe[feature_name])
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(dataframe.columns),
            fill_color='lightgray', align='center'),
        cells=dict(values=cells,
            fill_color='white',
            align='center'
            ))
        ])
          
        path_to_plotting_results = path_to_plotting_results
        fig.update_layout(barmode='group', xaxis_tickangle=-45,width=1500, title_text=task_type)
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        path = path_to_plotting_results / (task_type + "_plot.html")    
        fig.write_html(path)
        
