from Data.DatabaseAccessor.OpenMLAccessor import OpenMLAccessor
from Data.DatasetsStatistics.DatasetStatisticsCreator import DatasetStatisticsCreator
from Data.DatasetsStatistics.FeatureAnalyzer import FeatureAnalyzer
from Experiments.ExperimentReader import ExperimentReader
from PreprocessingPipeline.CategoricalFeaturePreprocessor import CategoricalFeaturePreprocessor
from utils import get_project_root



project_root = get_project_root()
path_to_openml_datasets = "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand"
path_experiment = project_root / "src/Experiments/ExperimentFiles/experiment_openml_jeager_by_hand.yaml"
path_dataset_statistic = project_root / "src/Data/DatasetsStatistics/openml_statistics_from_jaeger_datasets_by_hand" 
path_datasets_parent = project_root / path_to_openml_datasets


experiment_Reader = ExperimentReader(path_experiment)
experiment_config = experiment_Reader.get_experiment_config()
categorical_feature_encoder_names = experiment_config["preprocessing"]["categorical"]
dataset_statistic_creator = DatasetStatisticsCreator()

data = OpenMLAccessor.get_data_from_source(42712)

dataframe = data["data"]
feature = dataframe["weekday"]

value_collection = []
for index in range(dataframe.index.size):
    value = feature.loc[index]
    if value not in value_collection:
        value_collection.append(value)
print(str(value_collection))