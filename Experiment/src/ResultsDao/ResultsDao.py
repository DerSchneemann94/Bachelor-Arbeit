import json
from pathlib import Path
from typing import List
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl


class ResultsDao:
    @staticmethod
    def safe_results(path: Path, result):
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            performance = result["performance"] 
            performance.to_csv(path / f"downstream_performance_mean.csv")
            elapsed_time = result["elapsed_train_time"]
            experiment_config = result["experiment_config"]
            model_hyperparameter = result["model_hyperparameter"]
            model_hyperparameter = {
                'learner__loss': model_hyperparameter["learner__loss"],
                'learner__penalty': model_hyperparameter['learner__penalty'],
                'learner__alpha': model_hyperparameter["learner__alpha"]
            }
            DatasetStatisticDaoImpl.write_dataframe_statistic_to_file(path / "result.joblib", performance)
            DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "experiment.json", {"elapsed_time":elapsed_time,
                                                                                            "experiment_config":experiment_config,
                                                                                            "model_hyperparameter": model_hyperparameter})


    @staticmethod
    def safe_encoder_configuration_results_to_file(path: Path, encoder_configuration: List):
        encoder_configuration_json = {}
        combination_number = 1
        for configuration in encoder_configuration:
            encoder_configuration_json["enoder_combination_" + str(combination_number)] = configuration
            combination_number += 1    
        DatasetStatisticDaoImpl.write_json_statistic_to_file(path, encoder_configuration_json)    