import json
from datetime import datetime
import os
from pathlib import Path
from typing import List
from Data.Datasets_internal.PandasDataFrameCreator import PandaDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher
from Experiments.CategoricalFeatureEncodingExperiment_2 import CategoricalFeatureEncodingExperiment_2
from jenga.tasks.openml import OpenMLBinaryClassificationTask, OpenMLMultiClassClassificationTask, OpenMLRegressionTask
from utils import get_project_root

timestamp = "2023-03-17_15.00"
project_root = get_project_root()
path_to_openml_datasets = "src/Data/categorical_datasets_from_data_imputation_paper"
datasets_parent_path = project_root / path_to_openml_datasets
dataset_path = PathSearcher.get_path_of_datasets_with_timestamp_if_possible(datasets_parent_path, timestamp)
# task_dict = {
#     "Regression":ExternalDataMLRegressionTask,
#     "Binary-Classification":ExternalDataBinaryClassificationTask,
#     "Multiple-Classification":ExternalDataMultiClassClassificationTask
# }

task_dict = {
    "Regression":OpenMLRegressionTask,
    "Binary-Classification":OpenMLBinaryClassificationTask,
    "Multiple-Classification":OpenMLMultiClassClassificationTask
}


categorical_feature_encoder_names = ["ordinal_encode","one_hot_encode"]
timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M")
strategies = ["single_single"]
number_of_repetions = 2
numerical_feature_encoder_name="scaling"

if __name__ == "__main__":
    datasets_causing_exceptions = []
    for task_type in task_dict.keys():
        paths_to_data_set_csv = PathSearcher.get_list_of_dataset_paths(dataset_path / task_type)
        for path in paths_to_data_set_csv:
            task_class = task_dict[task_type]
            filename = os.path.split(path)[1]
            openml_id = filename.split(".")[0]
            dataframe = PandaDataFrameCreator.generate_dataframe_from_path(path=path)
            target_feature = dataframe.columns[-1]
            data, labels = PandaDataFrameCreator.split_dataframe_into_data_and_labels(dataframe=dataframe, target=target_feature)
            experiment = CategoricalFeatureEncodingExperiment_2(
                    task_id_class_tuples = [[openml_id, task_class]],
                    data=data,
                    labels=labels,
                    strategies=strategies,
                    num_repetitions=number_of_repetions,
                    categorical_feature_encoder_names=categorical_feature_encoder_names,
                    numerical_feature_encoder_name=numerical_feature_encoder_name,
                    timestamp="experiment_" + "_" + str(openml_id),
                    base_path="results/" + timestamp + "/" + task_type + "/" + openml_id
                    )
            try:   
                experiment.run() 
            except Exception as error:
                datasets_causing_exceptions.append(openml_id)
                continue

if datasets_causing_exceptions: 
    path = project_root / "results" / timestamp / "corrupted_datasets.txt"
    with open(path, "w") as file:
        file.write(str(datasets_causing_exceptions))


   





