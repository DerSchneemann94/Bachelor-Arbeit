import copy
import os
import time
from ResultsDao.ResultsDao import ResultsDao
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
from Experiments.ExperimentFeatureEncoding import ExperimentFeatureEncoding
from Experiments.ExperimentReader import ExperimentReader
from PreprocessingPipeline.CategoricalFeaturePreprocessor import CategoricalFeaturePreprocessor
from Data.DatabaseAccessor.OpenMLAccessor import OpenMLAccessor
from Data.Datasets_internal.PathSearcher import PathSearcher
from utils import get_project_root
from datetime import datetime
from jenga.tasks.ExternalDataTask import ExternalDataMLRegressionTask, ExternalDataBinaryClassificationTask, ExternalDataMultipleClassificationTask


project_root = get_project_root()
path_to_openml_datasets = "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand"
path_experiment = project_root / "src/Experiments/ExperimentFiles/experiment_openml_jeager_by_hand_specific.yaml"
path_dataset_statistic = project_root / "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand" 
path_datasets_parent = project_root / path_to_openml_datasets

task_dict = {
    "Regression":ExternalDataMLRegressionTask,
    "Binary-Classification":ExternalDataBinaryClassificationTask,
    "Multiple-Classification":ExternalDataMultipleClassificationTask
}

dataset_exceptions = ["4135", "42493"]

experiment_Reader = ExperimentReader(path_experiment)
experiment_config = experiment_Reader.get_experiment_config()
categorical_feature_encoder_names = experiment_config["preprocessing"]["categorical"]

experiment_name = datetime.now().strftime("%Y-%m-%d_%H.%M")
number_of_repetions = 2
numerical_feature_encoder_name="scaling"


if __name__ == "__main__":
    datasets_causing_exceptions = []
    for task_type in task_dict.keys():
        paths_to_data_set_csv = PathSearcher.get_list_of_dataset_paths(path_datasets_parent / task_type, "*.json")
        for path in paths_to_data_set_csv:
            task_class = task_dict[task_type]
            filename = os.path.split(path)[1]
            openml_id = filename.split("_")[0]
            if openml_id in dataset_exceptions:
                continue
            dataframe = OpenMLAccessor.get_data_from_source(openml_id)
            dataset_characteristic = DatasetStatisticsCreator.create_dataset_statistic_from_file(path_dataset_statistic / task_type / (str(openml_id) + "_characteristics.json"))
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
            experiment_encoder_combination_log = []
            try:  
                while True:
                    start_time = time.time()
                    if preprocessor.check_processor_state() is None:
                        break
                    experiment_encoder_combination = preprocessor.get_encoder_combination()
                    experiment_encoder_combination_log.append(experiment_encoder_combination)
                    experiment_config_copy = copy.deepcopy(experiment_config)
                    experiment_config_copy["preprocessing"]["categorical"] = experiment_encoder_combination
                    dataframe_transformed = preprocessor.transformData(data,True)
                    preprocessor.increment_processor_state()
                    elapsed_time = time.time() - start_time
                    experiment = ExperimentFeatureEncoding(
                        dataset_characteristic=dataset_characteristic,
                        original_data=data,
                        experiment_configuration=experiment_config_copy["preprocessing"],
                        elapsed_time=elapsed_time,    
                        task_id_class_tuples = [[openml_id, task_class]],
                        encoded_data=dataframe_transformed,
                        labels=labels,
                        num_repetitions=number_of_repetions,
                        experiment_name="experiment" + "_" + str(openml_id),
                        base_path=base_path,
                        seed=42
                    )
                    result = experiment.run()
                    ResultsDao.safe_results(base_path / ("encoder_combination_" + str(len(experiment_encoder_combination_log))), result)
                   
            except Exception as error:
                    datasets_causing_exceptions.append(openml_id)
                    raise error   
            
            instances = data.index.size
            instances_dict = {"instances": instances}
            path = project_root / ("results/" + experiment_name + "/" + task_type + "/" + str(openml_id)) 
            ResultsDao.safe_encoder_configuration_results_to_file(path / "encoder_combination_experiments_overview.json", experiment_encoder_combination_log)
            DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "experiment_config_file.json", experiment_config)
            DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "feature_characteristic.json", result["dataset_characteristic"])
            DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "dataset_characteristic.json", instances_dict)


if datasets_causing_exceptions: 
    path = project_root / "results" / experiment_name / "corrupted_datasets.txt"
    with open(path, "w") as file:
        file.write(str(datasets_causing_exceptions))


   





