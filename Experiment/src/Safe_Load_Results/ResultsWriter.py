from pathlib import Path
from typing import List
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl


class ResultsWriter:
    @staticmethod
    def safe_results(path: Path, result):
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            #Mean results
            performance = result["performance"] 
            performance.to_csv(path / f"downstream_performance_mean.csv")
            #safe metadata
            elapsed_time = result["elapsed_train_time"]
            experiment_config = result["experiment_config"]
            DatasetStatisticDaoImpl.write_dataframe_statistic_to_file(path / "result.joblib", performance)
            DatasetStatisticDaoImpl.write_json_statistic_to_file(path / "experiment.json", {"elapsed_time":elapsed_time, "experiment_config":experiment_config})
           
                        
            # results_path = path / "report for every experiment"
            # results_path.mkdir(parents=True, exist_ok=True)    
            # for index, (performance_data_frame, elapsed_train_time) in enumerate(
            #     zip(
            #         self._result.downstream_performances,
            #         self._result.elapsed_train_times                    
            #     )
            # ):
            #     performance_data_frame.to_csv(results_path / f"downstream_performance_rep_{index}.csv")
            #     Path(results_path / f"elapsed_train_time_rep_{index}.json").write_text(json.dumps(elapsed_train_time))


    @staticmethod
    def safe_encoder_configuration_results_to_file(path: Path, encoder_configuration: List):
        encoder_configuration_json = {}
        combination_number = 1
        for configuration in encoder_configuration:
            encoder_configuration_json["enoder_combination_" + str(combination_number)] = configuration
            combination_number += 1    
        DatasetStatisticDaoImpl.write_json_statistic_to_file(path, encoder_configuration_json)    