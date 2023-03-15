import json
from datetime import datetime
from pathlib import Path
from jenga.tasks.ExternalDataTask import ExternalDataBinaryClassificationTask, ExternalDataMultiClassClassificationTask, ExternalDataMLRegressionTask
from utils import get_project_root

project_root = get_project_root()
path = project_root / "src/Data/Datasets/DatasetsToAnalyze/datasets_that_are_used_list.txt"
    
tasks = [
    ExternalDataMLRegressionTask,
    ExternalDataBinaryClassificationTask,
    ExternalDataMultiClassClassificationTask
]
task_index = {
    "Regression":0,
    "Binary-Classification":1,
    "Multiple-Classification":2
}

datasets = json.loads(path.read_text())
categorical_feature_encoder_name = "ordinal_encode"
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
strategies = ["single_single"]
number_of_repetions = 2
numerical_feature_encoder_name="scaling"

datasets_causing_exceptions = []
for key in datasets.keys():
    dataset_data = datasets[key]
    task_type = dataset_data["type"]
    index = task_index[task_type]
    task_class = tasks[index]
    task_name = key

#     for openml_id in dataset_dictionary.keys():
#         experiment = CategoricalFeatureEncodingExperiment(
#             task_id_class_tuples = [[task_name, task]],
#             strategies=strategies,
#             num_repetitions=number_of_repetions,
#             categorical_feature_encoder_name=categorical_feature_encoder_name,
#             numerical_feature_encoder_name=numerical_feature_encoder_name,
#             timestamp="experiment_" + categorical_feature_encoder_name + "_" + str(openml_id),
#             base_path="results/" + timestamp + "/" + task_name
#             )
#         try:   
#             experiment.run() 
#         except:
#             datasets_causing_exceptions.append(openml_id)
#             continue

# if not openml_id: 
#     path = project_root / "results/" + timestamp + "/corrupted_datasets.txt"
#     with open(path, "w") as file:
#         file.write(datasets_causing_exceptions)

