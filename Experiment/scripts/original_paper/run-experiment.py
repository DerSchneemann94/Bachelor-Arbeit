
import json
from pathlib import Path
from typing import List, Tuple
from Experiments.CategoricalFeatureEncodingExperiment import CategoricalFeatureEncodingExperiment
from jenga.src.jenga.tasks.openml import OpenMLBinaryClassificationTask, OpenMLMultiClassClassificationTask, OpenMLRegressionTask, OpenMLTask
import typer



binary_task_id_mappings = json.loads(Path("../data/raw/binary.txt").read_text())
multi_task_id_mappings = json.loads(Path("../data/raw/multi.txt").read_text())
regression_task_id_mappings = json.loads(Path("../data/raw/regression.txt").read_text())
task_id_mappings = {**binary_task_id_mappings, **multi_task_id_mappings, **regression_task_id_mappings}

BINARY_TASK_IDS = [int(x) for x in binary_task_id_mappings.keys()]
MULTI_TASK_IDS = [int(x) for x in multi_task_id_mappings.keys()]
REGRESSION_TASK_IDS = [int(x) for x in regression_task_id_mappings.keys()]



def get_strategies(strategies) -> List[str]:
    return_value = [str(x) for x in strategies.lower().split(",")]

    for val in return_value:
        if val not in ["single_single", "multiple_multiple", "single_all", "multiple_all"]:
            raise ValueError(f"'{val}' is not a valid strategies")

    return return_value


def get_id_imputer_class_tuple(task_id: int) -> Tuple[int, OpenMLTask]:

    if task_id in BINARY_TASK_IDS:
        task_class = OpenMLBinaryClassificationTask

    elif task_id in MULTI_TASK_IDS:
        task_class = OpenMLMultiClassClassificationTask

    elif task_id in REGRESSION_TASK_IDS:
        task_class = OpenMLRegressionTask

    else:
        raise ValueError(f"task_id {task_id} isn't supported")

    return task_id, task_class



def main(
    task_id: int,
    experiment_name: str,
    categorical_encoder: str = "ordinal",
    numerical_encoder: str = "scaling",
    num_repetitions: int = 5,
    base_path: str = "/results",
    strategies: str = typer.Option(str, help="comma-seperated list")
    ):

    experiment_path = Path(base_path) / experiment_name / categorical_encoder / f"{task_id}"

    if experiment_path.exists():
        raise ValueError(f"Experiment at '{experiment_path}' already exist")

    task_id_class_tuples = [get_id_imputer_class_tuple(task_id=task_id)]
    experiment = CategoricalFeatureEncodingExperiment(
        task_id_class_tuples=task_id_class_tuples,
        num_repetitions=num_repetitions,
        base_path=base_path,
        categorical_feature_encoder_name = categorical_encoder,
        numerical_feature_encoder_name=numerical_encoder,
        strategies = get_strategies(strategies),
        timestamp=experiment_name,
    )
    experiment.run()


if __name__ == '__main__':
    typer.run(main)
