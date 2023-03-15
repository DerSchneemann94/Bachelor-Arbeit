import json
from datetime import datetime
from pathlib import Path
from Data.Datasets_internal.PandasDataFrameCreator import PandaDataFrameCreator
from Experiments.CategoricalFeatureEncodingExperiment2 import CategoricalFeatureEncodingExperiment2
from jenga.tasks.ExternalDataTask import ExternalDataBinaryClassificationTask, ExternalDataMultiClassClassificationTask, ExternalDataMLRegressionTask
from utils import get_project_root

project_root = get_project_root()
datasets_path = project_root / "src/Data/Datasets_identify/DatasetsToAnalyze/refractored_datasets"
datasets_metadata_path = project_root / "src/Data/Datasets_identify/DatasetsToAnalyze/datasets_that_are_used_list.txt"

task_dict = {
    "Regression":ExternalDataMLRegressionTask,
    "Binary-Classification":ExternalDataBinaryClassificationTask,
    "Multiple-Classification":ExternalDataMultiClassClassificationTask
}

datasets = json.loads(datasets_metadata_path.read_text())
categorical_feature_encoder_name = "ordinal_encode"
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
strategies = ["single_single"]
number_of_repetions = 2
numerical_feature_encoder_name="scaling"

datasets_causing_exceptions = []
for key in datasets.keys():
    dataset_metadata = datasets[key]
    task_type = dataset_metadata["type"]
    task_class = task_dict[task_type]
    task_name = key
    dataset_path = datasets_path / dataset_metadata["path"]
    dataframe = PandaDataFrameCreator.generate_dataframe_from_path(path=dataset_path)
    data, labels = PandaDataFrameCreator.split_dataframe_into_data_and_labels(dataframe=dataframe, target=dataset_metadata["target"])
    
    experiment = CategoricalFeatureEncodingExperiment2(
            task_id_class_tuples = [[task_name, task_class]],
            data=data,
            labels=labels,
            strategies=strategies,
            num_repetitions=number_of_repetions,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder_name=numerical_feature_encoder_name,
            timestamp="experiment_" + categorical_feature_encoder_name + "_" + str(task_name),
            base_path="results/" + timestamp + "/" + task_name
            )
    try:   
        experiment.run() 
    except:
        datasets_causing_exceptions.append(task_name)
        continue
   
