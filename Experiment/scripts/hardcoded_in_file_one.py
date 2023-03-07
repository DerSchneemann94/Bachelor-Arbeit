from pathlib import Path

from Experiments.CategoricalFeatureEncodingExperiment import CategoricalFeatureEncodingExperiment
from jenga.src.jenga.tasks.openml import OpenMLBinaryClassificationTask, OpenMLRegressionTask

experiment_path = Path("/results") / "experiment_debug" / "737"

if experiment_path.exists():
    raise ValueError(f"Experiment at '{experiment_path}' already exist")


experiment = CategoricalFeatureEncodingExperiment(
    task_id_class_tuples = [[42545, OpenMLRegressionTask]],
    strategies=["single_single"],
    num_repetitions=2,
    categorical_feature_encoder_name="ordinal_encode",
    numerical_feature_encoder_name="scaling",
    timestamp="experiment_regression"
)
            
experiment.run() 