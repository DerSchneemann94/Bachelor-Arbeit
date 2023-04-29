from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from ResultsEvaluator.ResultsEvaluator import ResultsEvaluator
from Visualization.GraphObjectFactory import GraphObjectFactory
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Visualization.VisualizationDataCreator import VisualizationDataCreator
from utils import get_project_root


root = get_project_root()
baseline = "one_hot"
path_to_plotting_results = root / "plot/time/detail/tables" / (baseline + "_without_multi")

experiment_mapping = {
    # "experiment_dummy":"dummy",
    # "experiment_dummy+ordinal+ordinal":"dummy+ordinal+ordinal",
    # "experiment_dummy+ordinal+cyclic":"dummy+ordinal+cyclic",
    "experiment_one_hot" : "one_hot",    
    # "experiment_one_hot+ordinal+cyclic": "one_hot+ordinal+cyclic",
    # "experiment_one_hot+ordinal+ordinal": "one_hot+ordinal+ordinal",
    # "experiment_ordinal" : "ordinal",
    # "experiment_hashing": "hashing",
    # "experiment_hashing+ordinal+ordinal": "hashing+ordinal+ordinal",
    # "experiment_hasing+ordinal+cyclic": "hasing+ordinal+cyclic",
    "experiment_catboost": "catboost",
    "experiment_catboost+ordinal+ordinal":"catboost+ordinal+ordinal",
    "experiment_catboost+ordinal+cyclic":"catboost+ordinal+cyclic",
    "experiment_glmm": "glmm",
    "experiment_glmm+ordinal+ordinal":"glmm+ordinal+ordinal", 
    "experiment_glmm+ordinal+cyclic":"glmm+ordinal+cyclic", 
    # "experiment_leave_one_out": "leave",
    # "experiment_leave_one_out+ordinal+ordinal": "leave+ordinal+ordinal",
    # "experiment_leave_one_out+ordinal+cyclic": "leave+ordinal+cyclic",
}

task_types = [
    "Binary-Classification",
    #"Multiple-Classification",
    "Regression"
]
unsuccessfull_datasets = []
if __name__ == "__main__":
    datasets_results = {}
    datasets_statistics_dataframes = {}
    datasets_results_combined = {}
    results_measured_time = {}
    results_dataframe = {}
    for task_type in task_types:
        results = {}
        for experiment_path in experiment_mapping.keys():
            path = root / "results" / "ridge+svm" / experiment_path / task_type
            result = VisualizationDataCreator.get_dataset_statistic(path)
            results_measured_time[experiment_path] = result    
        datasets_elapsed_time_dataframe = VisualizationDataCreator.create_elapsed_time_dataframe(results_measured_time, experiment_mapping)
        cells = []
        for feature_name in datasets_elapsed_time_dataframe.columns:
            cells.append(datasets_elapsed_time_dataframe[feature_name])
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(datasets_elapsed_time_dataframe.columns),
            fill_color='lightgray', align='center'),
        cells=dict(values=cells,
            fill_color='white',
            align='center'
            ),
            )
        ])
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        fig.update_layout(barmode='group', xaxis_tickangle=-45,width=2200, title_text=task_type)
        path = path_to_plotting_results / (task_type + "_plot.html")
        fig.write_html(path)