from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from ResultsEvaluator.ResultsEvaluator import ResultsEvaluator
from Visualization.GraphObjectFactory import GraphObjectFactory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Visualization.VisualizationDataCreator import VisualizationDataCreator
from utils import get_project_root


root = get_project_root()
baseline = "one_hot"
path_to_plotting_results = root / "plot/performance/bar_graph" / (baseline + "_without_multi")

experiment_mapping = {
    "experiment_dummy":"dummy",
    "experiment_dummy+ordinal+ordinal":"dummy+ordinal+ordinal",
    "experiment_dummy+ordinal+cyclic":"dummy+ordinal+cyclic",
    "experiment_one_hot" : "one_hot",    
    "experiment_one_hot+ordinal+cyclic": "one_hot+ordinal+cyclic",
    "experiment_one_hot+ordinal+ordinal": "one_hot+ordinal+ordinal",
    "experiment_ordinal" : "ordinal",
    "experiment_hashing": "hashing",
    "experiment_hashing+ordinal+ordinal": "hashing+ordinal+ordinal",
    "experiment_hasing+ordinal+cyclic": "hasing+ordinal+cyclic",
    # "experiment_catboost": "catboost",
    # "experiment_catboost+ordinal+ordinal":"catboost+ordinal+ordinal",
    # "experiment_catboost+ordinal+cyclic":"catboost+ordinal+cyclic",
    # "experiment_glmm": "glmm",
    # "experiment_glmm+ordinal+ordinal":"glmm+ordinal+ordinal", 
    # "experiment_glmm+ordinal+cyclic":"glmm+ordinal+cyclic", 
    "experiment_leave_one_out": "leave",
    "experiment_leave_one_out+ordinal+ordinal": "leave+ordinal+ordinal",
    "experiment_leave_one_out+ordinal+cyclic": "leave+ordinal+cyclic",
}

metrics = {
    "Binary-Classification": "F1_weighted",
    "Multiple-Classification": "F1_weighted",
    "Regression": "RMSE"
}

task_types = [
    "Binary-Classification",
    "Multiple-Classification",
    "Regression"
]


unsuccessfull_datasets = []


if __name__ == "__main__":
    datasets_results = {}
    datasets_statistics_dataframes = {}
    datasets_results_combined = {}
    results_dataframe = {}
    for task_type in task_types:
        results = {}
        for experiment_path in experiment_mapping.keys():
            path = root / "results" / "huber+log" / experiment_path / task_type
            result = VisualizationDataCreator.get_dataset_statistic(path)
            results[experiment_path] = VisualizationDataCreator.generate_dataframe_from_dataset_results(result)
        datasets_results_combined = VisualizationDataCreator.combine_dataframes_with_metric(results, metrics[task_type], experiment_mapping)
        encoder_performance_dataframe = datasets_results_combined.drop("openml_ids", axis=1)
        openml_ids = datasets_results_combined["openml_ids"]
        relative_performance_dataframe =  ResultsEvaluator.generate_relative_improvement_to_baseline_performance(encoder_performance_dataframe, baseline)
        fig = go.Figure(layout=go.Layout(
            title=go.layout.Title(text=task_type, font=dict(size=40)),
            xaxis = dict(
                title = 'openml_ids',
                showgrid = True,
                zeroline = True,
                showline = True,
                showticklabels = True,
                gridwidth = 1,
                titlefont = dict(
                    family = 'Arial, sans-serif',
                    size = 40,
                    color = 'lightgrey'
                ),
                tickfont = dict(
                family = 'Arial, sans-serif',
                size = 40,
                color = 'black'
                ),
            ),    
            legend=dict(
                font = dict(
                    family = "Arial, sans-serif",
                    size=40    
                )
            ),
            yaxis = dict(
                showgrid = True,
                zeroline = True,
                showline = True,
                title = 'relative performance',
                titlefont = dict(
                    family = 'Arial, sans-serif',
                    size = 40,
                    color = 'lightgrey'
                ),
                showticklabels = True,
                tickfont = dict(
                family = 'Arial, sans-serif',
                size = 40,
                color = 'black'
                ),
            ))
        )
        hovertemplate = ""
        graphfactory = GraphObjectFactory()
        for encoder in relative_performance_dataframe.columns:
            encoder_performance = relative_performance_dataframe[encoder]
            hovertemplate = "<b>performance: </b> %{y} <br>"
            hovertemplate += "<b>encoder: </b> %{meta[0]} <br>"
            fig.add_trace(graphfactory.create_bar_object(openml_ids, relative_performance_dataframe[encoder], encoder, [encoder])).update_traces(   
                marker={"line": {"width": 1, "color": "rgb(0,0,0)"}},
                hovertemplate=hovertemplate,
            )
        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        path = path_to_plotting_results / (task_type + "_plot.html")
        fig.write_html(path)
    
    
    