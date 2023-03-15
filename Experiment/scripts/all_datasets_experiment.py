import json
from datetime import datetime
from pathlib import Path
from Experiments.CategoricalFeatureEncodingExperiment import CategoricalFeatureEncodingExperiment
from jenga.tasks.openml import OpenMLBinaryClassificationTask, OpenMLMultiClassClassificationTask, OpenMLRegressionTask
from utils import get_project_root

project_root = get_project_root()
path = project_root / "data/raw"

tasks = [
    OpenMLBinaryClassificationTask,
    OpenMLMultiClassClassificationTask,
    OpenMLRegressionTask
]
task_types = [
    "Binary-Classifikation",
    "Multi-Classifikation",
    "Regression",
]
openml_datasets = [
    json.loads((path / "binary.txt").read_text()),
    json.loads((path / "multi.txt").read_text()),
    json.loads((path / "regression.txt").read_text())
]

categorical_feature_encoder_name = "ordinal_encode"
timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M")
strategies = ["single_single"]
number_of_repetions = 2
numerical_feature_encoder_name="scaling"

datasets_causing_exceptions = []
for index, dataset_dictionary in enumerate(openml_datasets):
    task = tasks[index]
    task_type = task_types[index]
    for openml_id in dataset_dictionary.keys():
        experiment = CategoricalFeatureEncodingExperiment(
            task_id_class_tuples = [[openml_id, task]],
            strategies=strategies,
            num_repetitions=number_of_repetions,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder_name=numerical_feature_encoder_name,
            timestamp="experiment_" + categorical_feature_encoder_name + "_" + str(openml_id),
            base_path="results/" + timestamp + "/" + task_type
            )
        try:   
            experiment.run() 
        except:
            datasets_causing_exceptions.append(openml_id)
            continue

if not openml_id: 
    path = project_root / "results/" + timestamp + "/corrupted_datasets.txt"
    with open(path, "w") as file:
        file.write(datasets_causing_exceptions)

