from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
import pandas as pd
import plotly.graph_objects as go
from Visualization.GraphObjectFactory import GraphObjectFactory
from Visualization.VisualizationDataCreator import VisualizationDataCreator
from utils import get_project_root

results_timestamp = "2023-04-13_11.04"
root = get_project_root()
path_to_results = root / "results" / results_timestamp
path_to_plotting_results = root / "plot/characteristics"


task_types = [
    "Binary-Classification",
    "Multiple-Classification",
    "Regression"
]

unsuccessfull_datasets = []

if __name__ == "__main__":
    datasets_statistics = {}
    datasets_statistics_dataframes = {}
    for task_type in task_types:
        datasets_statistics[task_type] = VisualizationDataCreator.get_dataset_statistic(path_to_results, task_type)
        datasets_statistics_dataframes[task_type] = DatasetStatisticsCreator.generate_dataframe_statistic_from_dataset_statistic(datasets_statistics[task_type])
    
    
    for task_type in datasets_statistics_dataframes.keys():        
        dataframe: pd.DataFrame = datasets_statistics_dataframes[task_type]
        fig = go.Figure()
        openml_ids = dataframe["openml_ids"]
        dataframe = dataframe.drop("openml_ids", axis=1)
        graphfactory = GraphObjectFactory()
        for data_type in dataframe:
            fig.add_trace(graphfactory.create_bar_object(openml_ids, dataframe[data_type], data_type)).update_traces(   
            marker={"line": {"width": 2, "color": "rgb(0,0,0)"}}
        )
        
        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        
        path_to_plotting_results = path_to_plotting_results / results_timestamp
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        path = path_to_plotting_results / (task_type + "_plot.html")    
        fig.write_html(path)


        
