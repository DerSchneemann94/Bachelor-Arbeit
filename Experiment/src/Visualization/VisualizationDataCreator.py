import os
from Visualization.Exceptions.NameMappingError import NameMappingError
from anyio import Path
import pandas as pd
from Data.DatasetsStatistics.DatasetStatisticDao.DatasetStatisticDaoImpl import DatasetStatisticDaoImpl
from Data.Datasets_internal.PandasDataFrameCreator import PandasDataFrameCreator
from Data.Datasets_internal.PathSearcher import PathSearcher


class VisualizationDataCreator:

    @staticmethod
    def get_dataset_statistic(results_path):
        datasets_statistics = {}
        #paths_to_datasets_results = PathSearcher.get_list_of_dataset_paths(results_path / task_type, "*_mean.csv")
        openml_ids = PathSearcher.get_list_of_subdirectories(results_path)
        for openml_id in openml_ids:
            path_to_dataset_result = PathSearcher. get_list_of_dataset_paths(results_path / openml_id, "*_mean.csv")[0]
            path_dataset_experiment = Path(os.path.split(path_to_dataset_result)[0])
            experiment_result = PandasDataFrameCreator.generate_dataframe_from_path(path_to_dataset_result)
            experiment_result = experiment_result.set_index(experiment_result.columns[0])
            experiment_setup = DatasetStatisticDaoImpl.read_statistic_from_json(path_dataset_experiment / "experiment.json")
            encoder_setup = experiment_setup
            path_to_dataset_statistic = Path(os.path.split(path_dataset_experiment)[0])
            dataset_characteristic = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_dataset_statistic / "dataset_characteristic.json")
            feature_characteristic = DatasetStatisticDaoImpl.read_statistic_from_json(path_to_dataset_statistic / "feature_characteristic.json")
            statisitic = {
                "experiment_result": experiment_result,
                "dataset_characteristic": dataset_characteristic,
                "encoder_setup": encoder_setup,
                "feature_characteristic": feature_characteristic
            }
            datasets_statistics[openml_id]= statisitic
        return datasets_statistics
       
    
    @staticmethod
    def generate_dataframe_from_dataset_results(datasets_results):
        openml_ids = []
        performance_dictionary = {}
        openml_id = list(datasets_results.keys())[0]
        metrics = datasets_results[openml_id]["experiment_result"].index
        for metric in metrics:
            performance_dictionary[metric] = []
        for openml_id in datasets_results.keys():
            try:
                results:pd.DataFrame = datasets_results[openml_id]["experiment_result"]
                for metric in results.index: 
                    performance_dictionary[metric].append(results.loc[metric]["performance"])
                openml_ids.append(openml_id)     
            except Exception as error:
                continue
        buffer = {"openml_ids":openml_ids}
        performance_dictionary = {**buffer, **performance_dictionary}

        performance_dataframe = pd.DataFrame(performance_dictionary)    
        return performance_dataframe
    

    @staticmethod
    def combine_dataframes_with_metric(results, metric, name_mapping):
        buffer = results[list(results.keys())[0]]
        openml_ids = buffer["openml_ids"]
        combined_results_dict = {}
        for result_dataframe in results.keys():
            dataframe = results[result_dataframe]
            values = dataframe[metric]
            try:
                result_name = name_mapping[result_dataframe]
            except:
                raise NameMappingError(result_name + " is not available in name_mapping.")
            combined_results_dict[result_name] = values
        combined_results_dict = {**{"openml_ids":openml_ids}, **combined_results_dict}    
        dataframe = pd.DataFrame(combined_results_dict)
        return dataframe


