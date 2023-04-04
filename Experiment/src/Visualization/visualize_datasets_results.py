import json
import math
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.DatasetsStatistics.DatasetStatisticTransformer import DatasetStatisticTransformer
from ResultsEvaluator.ResultsEvaluator import ResultsEvaluator
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from utils import get_project_root

results_timestamp = "2023-03-31_12.51"
root = get_project_root()
path_to_datasets_statistic = root / "src/Data/Datasets_statistics/openml_statistics_from_jaeger_datasets_ordinal_by_hand"
path_to_results = root / "results" / results_timestamp
path_to_plotting_results = root / "plot"



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

colors = [
                "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
                "beige", "bisque", "black", "blanchedalmond", "blue",
                "blueviolet", "brown", "burlywood", "cadetblue",
                "chartreuse", "chocolate", "coral", "cornflowerblue",
                "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
                "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
                "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                "darkorchid", "darkred", "darksalmon", "darkseagreen",
                "darkslateblue", "darkslategray", "darkslategrey",
                "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
                "dimgray", "dimgrey", "dodgerblue", "firebrick",
                "floralwhite", "forestgreen", "fuchsia", "gainsboro",
                "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
                "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
                "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
                "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
                "lightgoldenrodyellow", "lightgray", "lightgrey",
                "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
                "lightskyblue", "lightslategray", "lightslategrey",
                "lightsteelblue", "lightyellow", "lime", "limegreen",
                "linen", "magenta", "maroon", "mediumaquamarine",
                "mediumblue", "mediumorchid", "mediumpurple",
                "mediumseagreen", "mediumslateblue", "mediumspringgreen",
                "mediumturquoise", "mediumvioletred", "midnightblue",
                "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
                "oldlace", "olive", "olivedrab", "orange", "orangered",
                "orchid", "palegoldenrod", "palegreen", "paleturquoise",
                "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
                "plum", "powderblue", "purple", "red", "rosybrown",
                "royalblue", "saddlebrown", "salmon", "sandybrown",
                "seagreen", "seashell", "sienna", "silver", "skyblue",
                "slateblue", "slategray", "slategrey", "snow", "springgreen",
                "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
                "violet", "wheat", "white", "whitesmoke", "yellow",
                "yellowgreen"
]

unsuccessfull_datasets = []


def get_datset_statistic(results_path, task_type):
    datasets_statistics = {}
    #paths_to_datasets_results = PathSearcher.get_list_of_dataset_paths(results_path / task_type, "*_mean.csv")
    openml_ids = PathSearcher.get_list_of_subdirectories(results_path / task_type)
    for openml_id in openml_ids:
        path_to_dataset_results = PathSearcher. get_list_of_dataset_paths(results_path / task_type / openml_id, "*_mean.csv")
        result = PandasDataFrameCreator.generate_dataframe_from_paths(path_to_dataset_results)
        statisitic = {
            "results": result,        
        }
        datasets_statistics[openml_id]= statisitic
    return datasets_statistics


if __name__ == "__main__":
    datasets_statistics = {}
    datasets_statistics_dataframes = {}
    for task_type in task_types:
        datasets_statistics[task_type] = get_datset_statistic(path_to_results, task_type)
    for task_type in datasets_statistics.keys():
        results = datasets_statistics[task_type]
        columns = 5 
        rows = math.ceil(len(results.keys()) / columns)
        subplot = make_subplots(rows=rows, cols=columns)
        metric = metrics[task_type]
        dataframe = PandasDataFrameCreator.generate_dataframe_from_dataset_results(results, metric)
        plot_number = 1
        encoder_performance_dataframe = dataframe.drop("openml_ids", axis=1)
        encoder_performance_dataframe_relative_to_baseline = ResultsEvaluator.generate_relative_improvement_to_baseline_performance(encoder_performance_dataframe, "ordinal")
        openml_ids = dataframe["openml_ids"]
        fig = go.Figure( layout=go.Layout(
            title=go.layout.Title(text=task_type))
        )
        color_number = 0

        for encoder in encoder_performance_dataframe_relative_to_baseline.keys():
            single_encoder_performance = encoder_performance_dataframe_relative_to_baseline[encoder]
            hovertemplate = "<b>performance: </b> %{y} <br>"
            hovertemplate += "<b>encoder: </b> %{meta[0]} <br>"
            colors = []
            for value in single_encoder_performance:
                color = 'green' if value >= 0 else 'red'
                colors.append(color)
            fig.add_trace(go.Bar(
                    x=openml_ids,
                    y=single_encoder_performance,
                    name=encoder,
                    marker_color=colors,
                    meta=[encoder]

                )).update_traces(   
                    marker={"line": {"width": 1, "color": "rgb(0,0,0)"}},
                    hovertemplate=hovertemplate
                )

            color_number += 1
        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        path = path_to_plotting_results / (task_type + "_plot.html")
        fig.write_html(path)


       