
import json
from pathlib import Path
from typing import List, Tuple
from teilautomatischeEnkodierung.Experiments.Different_Categorial_Feature_Encoding import Experiment

import typer
from jenga.tasks.openml import (
    OpenMLBinaryClassificationTask,
    OpenMLMultiClassClassificationTask,
    OpenMLRegressionTask,
    OpenMLTask
)


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


def get_imputer_class_and_arguments(imputer_name: str) -> BaseImputer:

    if imputer_name not in IMPUTER_CLASS.keys():
        raise ValueError(f"imputer_name '{imputer_name}' not known. Choose one of: {', '.join(IMPUTER_CLASS.keys())}")

    return IMPUTER_CLASS[imputer_name], IMPUTER_ARGUMENTS[imputer_name], IMPUTER_NAME[imputer_name]


def main(
    task_id: int,
    experiment_name: str,
    encoding_strategy: str,
    machine_learning_model: str,
    num_repetitions: int = 5,
    base_path: str = "/results"
):

    experiment_path = Path(base_path) / experiment_name / encoding_strategy / f"{task_id}"

    if experiment_path.exists():
        raise ValueError(f"Experiment at '{experiment_path}' already exist")

    experiment = Experiment(
        task_id_class_tuples=[get_id_imputer_class_tuple(task_id)],
        num_repetitions=num_repetitions,
        encoding_strategies,
        machine_learning_model,
        base_path=base_path,
        timestamp=experiment_name,
        fully_observed=False if "corrupted" in experiment_name else True
    )
    experiment.run(task_id_mappings[f"{task_id}"])


if __name__ == '__main__':
    typer.run(main)
