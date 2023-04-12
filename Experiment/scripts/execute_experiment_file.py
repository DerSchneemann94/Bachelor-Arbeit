import copy
import os
from pathlib import Path
import time
import joblib
from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
from Experiments.ExperimentFeatureEncoding import ExperimentFeatureEncoding
from Experiments.ExperimentReader import ExperimentReader
from PreprocessingPipeline.CategoricalFeaturePreprocessor import CategoricalFeaturePreprocessor
from Data.DatabaseAccessor.OpenMLAccessor import OpenMLAccessor
from Data.Datasets_internal.PathSearcher import PathSearcher
from jenga.tasks.openml import OpenMLBinaryClassificationTask, OpenMLMultiClassClassificationTask, OpenMLRegressionTask
from utils import get_project_root
from datetime import datetime
from jenga.tasks.ExternalDataTask import ExternalDataTask, ExternalDataMLRegressionTask, ExternalDataBinaryClassificationTask, ExternalDataMultiClassClassificationTask


project_root = get_project_root()
path_to_openml_datasets = "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand"
path_experiment = project_root / "src/Experiments/ExperimentFiles/experiment_openml_jeager_by_hand.yaml"
path_dataset_statistic = project_root / "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand" 
path_datasets_parent = project_root / path_to_openml_datasets

task_dict = {
    "Regression":ExternalDataMLRegressionTask,
    "Binary-Classification":ExternalDataBinaryClassificationTask,
    "Multiple-Classification":ExternalDataMLRegressionTask
}

experiment_Reader = ExperimentReader(path_experiment)
experiment_config = experiment_Reader.get_experiment_config()
categorical_feature_encoder_names = experiment_config["preprocessing"]["categorical"]

experiment_name = datetime.now().strftime("%Y-%m-%d_%H.%M")
number_of_repetions = 2
numerical_feature_encoder_name="scaling"


if __name__ == "__main__":
    datasets_causing_exceptions = []
    dataset_statistic_creator = DatasetStatisticsCreator()
    for task_type in task_dict.keys():
        paths_to_data_set_csv = PathSearcher.get_list_of_dataset_paths(path_datasets_parent / task_type, "*.json")
        for path in paths_to_data_set_csv:
            task_type = "Binary-Classification"
            task_class = task_dict[task_type]
            filename = os.path.split(path)[1]
            #openml_id = filename.split("_")[0]
            openml_id = "4135"
            print("openml_id:  " + openml_id)
            dataframe = OpenMLAccessor.get_data_from_source(openml_id)
            dataset_characteristic = dataset_statistic_creator.create_dataset_statistic_from_file(path_dataset_statistic / task_type / (str(openml_id) + "_characteristics.json"))
            feature_statistic = FeatureAnalyzer.get_categorical_composition_of_dataframe(dataset_characteristic)
            preprocessor = CategoricalFeaturePreprocessor(categorical_feature_encoder_names, feature_statistic, dataset_characteristic)
            base_path = project_root / ("results/" + experiment_name + "/" + task_type + "/" + openml_id)
            if base_path.exists():
                raise ValueError(f"Experiment already exist")
            else:
                base_path.mkdir(parents=True, exist_ok=True)
            data = dataframe.data
            labels = dataframe.target
            dataframe_transformed = copy.deepcopy(data)
            try:  
                while True:
                    start_time = time.time()
                    dataframe_transformed = preprocessor.transformData(dataframe_transformed)
                    if dataframe_transformed is None:
                        break
                    elapsed_time = time.time() - start_time
                    experiment = ExperimentFeatureEncoding(
                        dataset_characteristic=dataset_characteristic,
                        original_data=data,
                        experiment_configuration=experiment_config,
                        elapsed_time=elapsed_time,    
                        task_id_class_tuples = [[openml_id, task_class]],
                        encoded_data=dataframe_transformed,
                        labels=labels,
                        num_repetitions=number_of_repetions,
                        experiment_name="experiment" + "_" + str(openml_id),
                        base_path=base_path,
                        seed=42
                    )
                    experiment.run()
            except Exception as error:
                    datasets_causing_exceptions.append(openml_id)
                    continue    





if datasets_causing_exceptions: 
    path = project_root / "results" / experiment_name / "corrupted_datasets.txt"
    with open(path, "w") as file:
        file.write(str(datasets_causing_exceptions))


   





