import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from utils import get_project_root


root = get_project_root()
path_to_datasets_statistic = root / "src/Data/Datasets_identify/openml_statistics"
results_timestamp = "2023-03-20_14.46"
path_to_results = root / "results"


task_types = [
    "Binary-Classification",
    "Multiple-Classification",
    "Regression"
]

unsuccessfull_datasets = []

def create_dataframe_for_task_type(task_type) -> pd.DataFrame:
        path = path_to_datasets_statistic / task_type
        dataset_statistic_paths = list(path.rglob('*.txt'))
        dataframes = []
        for path in dataset_statistic_paths:
            openml_id = extract_openml_id(path)
            try:
                dataset_characteristics = create_datasetcharacteristics_dataframe(path, openml_id)
                #dataset_performance = create_performance_dataframe(openml_id, task_type)
                dataframes.append(dataset_characteristics)
            except Exception as error:
                unsuccessfull_datasets.append(openml_id)
                raise error
                continue   
        dataframe = pd.concat(dataframes)
        return dataframe


def create_datasetcharacteristics_dataframe(path: Path, openml_id: str):
    with open(path, "r") as file:
        lines = file.readlines()
        columns = [
            lines[-3],
            lines[-4],
            lines[-1],
            lines[-2],
        ]
        data = create_column_dict(columns)
        buffer = {"openml_id" : openml_id}
        data = {**buffer, **data}
        dataframe = pd.DataFrame([data])
        return dataframe


def create_performance_dataframe(openml_id, task_type):
    path = path_to_results / results_timestamp / task_type / openml_id


def extract_openml_id(path: str):
    file = str(path).split("/")[-1]
    filename = file.split(".")[0]
    openml_id = filename.split("_")[0]
    return openml_id


def create_column_dict(strings: List[str]) -> Dict[str, str]:
    dict = {}
    for string in strings:
        string = string.replace("\n", "")
        key, value = string.split(":")
        value = value.replace(" ", "")
        print("test")
        dict[str(key)] = str(value)
    return dict


if __name__ == "__main__":
    data_collection = {}
    for task_type in task_types:
        data_collection[task_type] = create_dataframe_for_task_type(task_type)
    for task_type in data_collection.keys():
        dataframe = data_collection[task_type]
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(dataframe.columns),
            fill_color='paleturquoise', align='left'),
        cells=dict(values=[dataframe.openml_id, dataframe.instances, dataframe.features, dataframe.nominal_cardinality, dataframe.ordinal_cardinality],
            fill_color='lavender',
            align='left'))
        ])
        fig.show()   


        
