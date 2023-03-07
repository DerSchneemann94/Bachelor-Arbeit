from pathlib import Path

from Experiments.CategoricalFeatureEncodingExperiment import CategoricalFeatureEncodingExperiment
from jenga.src.jenga.tasks.openml import OpenMLBinaryClassificationTask

experiment_path = Path("/results") / "experiment_debug" / "737"

if experiment_path.exists():
    raise ValueError(f"Experiment at '{experiment_path}' already exist")


experiment = CategoricalFeatureEncodingExperiment(
    task_id_class_tuples = [[737, OpenMLBinaryClassificationTask]],
    strategies=["single_single"],
    num_repetitions=5,
    categorical_feature_encoder_name="one-hot",
    numerical_feature_encoder_name="scaling",
    timestamp="experiment_debug2"
)
            
experiment.run() 