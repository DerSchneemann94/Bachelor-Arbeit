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

experiment_mapping = {
    "experiment_catboost": "catboost",
    "experiment_catboost+ordinal+ordinal":"catboost+ordinal+ordinal",
    "experiment_catboost+ordinal+cyclic":"catboost+ordinal+cyclic",
    "experiment_glmm": "glmm",
    "experiment_glmm+ordinal+ordinal":"glmm+ordinal+ordinal", 
    "experiment_glmm+ordinal+cyclic":"glmm+ordinal+cyclic", 
    "experiment_ordinal" : "ordinal",
    "experiment_one_hot" : "one_hot",    
    "experiment_one_hot+ordinal+cyclic": "one_hot+ordinal+cyclic",
    "experiment_one_hot+ordinal+ordinal": "one_hot+ordinal+ordinal",
    "experiment_hashing": "hashing",
    "experiment_hashing+ordinal+ordinal": "hashing+ordinal+ordinal",
    "experiment_hasing+ordinal+cyclic": "hasing+ordinal+cyclic",
    "experiment_leave_one_out": "leave",
    "experiment_leave_one_out+ordinal+ordinal": "leave+ordinal+ordinal",
    "experiment_leave_one_out+ordinal+cyclic": "leave+ordinal+cyclic",
}

project_root = get_project_root()
experiments_root= "results/without_svm"
path_experiment = [
    {
        "path":project_root / "src/Experiments/ExperimentFiles/experiment_openml_jaeger_by_hand_2.yaml",
        "models": "ridge+svm"
    },
    {
        "path":project_root / "src/Experiments/ExperimentFiles/experiment_openml_jaeger_by_hand.yaml",
        "models": "huber+log"
    }                   
]
path_dataset_statistic = project_root / "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand" 

task_dict = {
    "Regression":ExternalDataMLRegressionTask,
    "Binary-Classification":ExternalDataBinaryClassificationTask,
    "Multiple-Classification":ExternalDataMultipleClassificationTask
}

dataset_exceptions = ["4135", "42493"]



if __name__ == "__main__":
    for model_experiment_configuration_file in path_experiment:
        experiment_Reader = ExperimentReader(model_experiment_configuration_file["path"])
        experiment_config = experiment_Reader.get_experiment_config()
        new_models_config = experiment_config["model"]
        for experiment in experiment_mapping.keys():
            experiments_root_path = project_root / experiments_root
            path_experiments = PathSearcher.get_list_of_subdirectories(experiments_root_path)
            for path_experiment in path_experiments:
                experiment_name = path_experiment
                if (project_root / "results" / model_experiment_configuration_file["models"] / experiment_name).exists():
                    continue
                path_experiment = experiments_root_path / path_experiment
                task_types = PathSearcher.get_list_of_subdirectories(path_experiment)
                for task_type in task_types:
                    datasets_causing_exceptions = [] 
                    task_class = task_dict[task_type]
                    path_experiment_by_task_type = path_experiment / task_type
                    openml_ids = PathSearcher.get_list_of_subdirectories(path_experiment_by_task_type)
                    for openml_id in openml_ids:
                        path_to_single_experiment = path_experiment_by_task_type / openml_id
                        feature_characteristics = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_single_experiment / "feature_characteristic.json")
                        if openml_id in dataset_exceptions:
                            continue
                        dataframe = OpenMLAccessor.get_data_from_source(openml_id)
                        feature_statistic = FeatureAnalyzer.get_categorical_composition_of_dataframe(feature_characteristics)
                        experiment_config = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_single_experiment / "experiment_config_file.json") 
                        experiment_config["model"] = new_models_config
                        categorical_feature_encoder_names = experiment_config["preprocessing"]["categorical"]
                        preprocessor = CategoricalFeaturePreprocessor(categorical_feature_encoder_names, feature_statistic, feature_characteristics)
                        base_path = project_root / ("results/" + model_experiment_configuration_file["models"] + "/" + experiment_name + "/" + task_type + "/" + openml_id)
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
                                try:
                                    model_name = experiment_config_copy["model"][task_type][0]
                                except:
                                    model_name = experiment_config_copy["model"][task_type.lower()][0]
                                dataframe_transformed = preprocessor.transformData(data,labels,True)
                                preprocessor.increment_processor_state()
                                elapsed_time = time.time() - start_time
                                experiment = ExperimentFeatureEncoding(
                                    dataset_characteristic=feature_characteristics,
                                    original_data=data,
                                    experiment_configuration=experiment_config_copy["preprocessing"],
                                    elapsed_time=elapsed_time,    
                                    task_id_class_tuples = [[openml_id, task_class]],
                                    encoded_data=dataframe_transformed,
                                    labels=labels,
                                    model_name=model_name,
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
                        path = base_path
                        ResultsDao.safe_encoder_configuration_results_to_file(path / "encoder_combination_experiments_overview.json", experiment_encoder_combination_log)
                        DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "experiment_config_file.json", experiment_config)
                        DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "feature_characteristic.json", result["dataset_characteristic"])
                        DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "dataset_characteristic.json", instances_dict)


if datasets_causing_exceptions: 
    path = project_root / "results" / experiment_name / "corrupted_datasets.txt"
    with open(path, "w") as file:
        file.write(str(datasets_causing_exceptions))


   






