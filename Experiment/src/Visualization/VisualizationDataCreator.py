import os
from anyio import Path
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher


class VisualizationDataCreator:

    @staticmethod
    def get_datset_statistic(results_path, task_type):
        datasets_statistics = {}
        #paths_to_datasets_results = PathSearcher.get_list_of_dataset_paths(results_path / task_type, "*_mean.csv")
        openml_ids = PathSearcher.get_list_of_subdirectories(results_path / task_type)
        for openml_id in openml_ids:
            path_to_dataset_result = PathSearcher. get_list_of_dataset_paths(results_path / task_type / openml_id, "*_mean.csv")[0]
            path_dataset_experiment = Path(os.path.split(path_to_dataset_result)[0])
            experiment_result = PandasDataFrameCreator.generate_dataframe_from_path(path_to_dataset_result)
            experiment_result = experiment_result.set_index(experiment_result.columns[0])
            experiment_setup = DatasetStatisticDaoImpl.read_statistic_from_json(path_dataset_experiment / "experiment.json")
            encoder_setup = experiment_setup
            path_to_dataset_statistic = Path(os.path.split(path_dataset_experiment)[0])
            dataset_characteristic = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_dataset_statistic / "dataset_characteristic.json")
            statisitic = {
                "experiment_result": experiment_result,
                "dataset_characteristic": dataset_characteristic,
                "encoder_setup": encoder_setup      
            }
            datasets_statistics[openml_id]= statisitic
        return datasets_statistics