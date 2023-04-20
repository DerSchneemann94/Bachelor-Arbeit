from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from ResultsEvaluator.ResultsEvaluator import ResultsEvaluator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Visualization.VisualizationDataCreator import VisualizationDataCreator
from utils import get_project_root


root = get_project_root()
baseline = "one_hot"
experiment_paths = ["experiment_one_hot", "experiment_ordinal"]
path_to_plotting_results = root / "plot/performance" / baseline

name_mapping = {
    "experiment_one_hot" : "one_hot",
    "experiment_ordinal" : "ordinal"
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
        for experiment_path in experiment_paths:
            path = root / "results" / experiment_path / task_type
            result = VisualizationDataCreator.get_dataset_statistic(path)
            results[experiment_path] = VisualizationDataCreator.generate_dataframe_from_dataset_results(result)
        datasets_results_combined = VisualizationDataCreator.combine_dataframes_with_metric(results, metrics[task_type], name_mapping)
        encoder_performance_dataframe = datasets_results_combined.drop("openml_ids", axis=1)
        openml_ids = datasets_results_combined["openml_ids"]
        relative_performance_dataframe =  ResultsEvaluator.generate_relative_improvement_to_baseline_performance(encoder_performance_dataframe, baseline)
        statistic_dataframe = DatasetStatisticsCreator.generate_dataframe_statistic_from_dataset_statistic(result)
        statistic_dataframe.drop(columns="openml_ids", inplace=True)
        fig = go.Figure( layout=go.Layout(
            title=go.layout.Title(text=task_type))
        )
        hovertemplate = ""
        counter = 1
        base_meta = []
        for data_type in statistic_dataframe.keys():
            hovertemplate +=  "<b>{data_type}_features: </b> %{meta[counter]} <br>"
            counter += 1
            base_meta.append(statistic_dataframe[data_type].values)
        for encoder in relative_performance_dataframe.columns:
            single_encoder_performance = relative_performance_dataframe[encoder]
            hovertemplate = "<b>performance: </b> %{y} <br>"
            hovertemplate += "<b>encoder: </b> %{meta[0]} <br>"
            colors = []
            base_meta = base_meta.insert(0, encoder)
            for value in single_encoder_performance:
                color = 'green' if value >= 0 else 'red'
                colors.append(color)
            fig.add_trace(go.Bar(
                    x=openml_ids,
                    y=single_encoder_performance,
                    name=encoder,
                    marker_color=colors,
                    meta=base_meta
                )).update_traces(   
                    marker={"line": {"width": 1, "color": "rgb(0,0,0)"}},
                    hovertemplate=hovertemplate
                )
        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        if not path_to_plotting_results.exists():
            path_to_plotting_results.mkdir(parents=True, exist_ok=True)
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        path = path_to_plotting_results / (task_type + "_plot.html")
        fig.write_html(path)