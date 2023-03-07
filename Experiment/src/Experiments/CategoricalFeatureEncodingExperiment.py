import json
import logging
import os
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from jenga.src.jenga.preprocessing.categorical_feature_encoder.CategoricalEncoder import Encoder
from jenga.src.jenga.tasks.openml import OpenMLTask

import joblib
import pandas as pd
from Evaluation.Evaluation import EvaluationResult, MultipleColumnsAllMissingEvaluator, MultipleColumnsEvaluator, SingleColumnAllMissingEvaluator, SingleColumnEvaluator
from utils import get_project_root


logger = logging.getLogger()


class CategoricalFeatureEncodingExperiment(object):

    def __init__(
        self,
        task_id_class_tuples: List[Tuple[int, Callable[..., OpenMLTask]]],
        strategies: List[str],
        num_repetitions: int,
        numerical_feature_encoder_name = "scaling",
        categorical_feature_encoder_name = "ordinal_encode",
        base_path: str = 'results',
        timestamp: Optional[str] = None,
    ):

        self.strategy_to_EvaluatorClass = {
            "single_single": SingleColumnEvaluator,
            "multiple_multiple": MultipleColumnsEvaluator,
            "single_all": SingleColumnAllMissingEvaluator,
            "multiple_all": MultipleColumnsAllMissingEvaluator
        }

        self._task_id_class_tuples = task_id_class_tuples
        self._strategies = strategies
        self._num_repetitions = num_repetitions
        self._categorical_feature_encoder_name = categorical_feature_encoder_name
        self._categorical_feature_encoder = Encoder.getCategoricalEncoder(self._categorical_feature_encoder_name)   
        self._numerical_feature_encoder_name = numerical_feature_encoder_name
        self._numerical_feature_encoder = Encoder.getNumericalEncoder(self._numerical_feature_encoder_name) 
        self._timestamp = timestamp
        self._result: Dict[int, Dict[str, Dict[float, Dict[str, EvaluationResult]]]] = dict()

        valid_strategies = self.strategy_to_EvaluatorClass.keys()
        for strategy in self._strategies:
            if strategy not in valid_strategies:
                raise Exception(f"'{strategy}' is not a valid strategy. Need to be in {', '.join(valid_strategies)}")

        project_root = get_project_root()
        self._base_path = project_root / base_path

        if self._timestamp is None:
            self._timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        experiment_path = self._base_path / timestamp
        if experiment_path.exists():
            raise ValueError(f"Experiment already exist")

        self._base_path = self._base_path / self._timestamp / self._categorical_feature_encoder_name
        
        


    def run(self):
        for task_id, task_class in self._task_id_class_tuples:
            self._result[task_id] = {}
            task = task_class(openml_id=task_id)
            for strategy in self._strategies:
                experiment_path = self._base_path / f"{task_id}" / f"{strategy}"
                try:
                    evaluator = self.strategy_to_EvaluatorClass[strategy](
                        task=task,
                        path=experiment_path,
                        categorical_feature_encoder=self._categorical_feature_encoder,
                        categorical_feature_encoder_name=self._categorical_feature_encoder_name,
                        numerical_feature_encoder=self._numerical_feature_encoder,
                        numerical_feature_encoder_name=self._numerical_feature_encoder_name
                        )
                    evaluator.evaluate(self._num_repetitions)
                    result = evaluator._result
                except Exception:
                    error = traceback.format_exc()
                    experiment_path.mkdir(parents=True, exist_ok=True)
                    Path(experiment_path / "error.txt").write_text(str(error))
                    logger.exception(f"Tried to use {self._categorical_feature_encoder.__name__}")
                    result = error

                    self._result[task_id][strategy] = result

        joblib.dump(self._result, Path(self._base_path / f"{task_id}" / "result.joblib"))
        Path(self._base_path / f"{task_id}" / "evaluation_parameters.json").write_text(
            json.dumps(
                {
                    "encoder_name": self._categorical_feature_encoder.__name__,
                    "strategy": self._strategies
                }
            )
        )

        logger.info(f"Experiment Finished! - Results are at: {self._base_path.parent}")


def _recursive_split(path):
    """
    Recursively splits a path into its components.

    Returns:
        tuple
    """
    rest, tail = os.path.split(path)
    if rest in ('', os.path.sep):
        return tail,
    return _recursive_split(rest) + (tail,)


def read_experiment(path):
    """
    Discovers CSV files an experiment produced and construct columns
    for the experiment's conditions from the sub-directory structure.

    Args:
        path: path to the experiment's results.

    Returns:
        pd.DataFrame
    """
    objects = list(path.rglob('*.csv'))
    data = []
    path_split = _recursive_split(path)

    for obj in objects:
        obj_path_split = _recursive_split(obj)
        if len(obj_path_split) - len(path_split) > 7:
            raise Exception("Path depth too long! Provide path to actual experiment or one of its sub-directories.")
        data.append(obj_path_split)

    df = pd.DataFrame(data=data)

    columns = ["experiment", "imputer", "task", "missing_type", "missing_fraction", "strategy", "file_or_dir", "detail_file"]
    auto_columns = []
    for i in range(df.shape[1] - len(columns)):
        auto_columns.append(f"col{i}")
    df.columns = auto_columns + columns
    df.drop(auto_columns, axis=1, inplace=True)

    df["path"] = objects
    df["detail_file"] = df["detail_file"].fillna("")

    return df.reset_index(drop=True)


def _read_prefixed_csv_files(df_experiment, file_prefix, read_details):
    col_pattern = f"({file_prefix}_)(\\S*)(.csv)"
    dfs = []
    if read_details:
        file_col = "detail_file"
    else:
        file_col = "file_or_dir"
    # TODO this loop is pretty slow
    for row in df_experiment[df_experiment[file_col].str.startswith(file_prefix)].iterrows():
        df_new = pd.read_csv(row[1]["path"])
        df_new.rename({"Unnamed: 0": "metric"}, inplace=True, axis=1)
        df_new["experiment"] = row[1]["experiment"]
        df_new["imputer"] = row[1]["imputer"]
        df_new["task"] = row[1]["task"]
        df_new["missing_type"] = row[1]["missing_type"]
        df_new["missing_fraction"] = row[1]["missing_fraction"]
        df_new["strategy"] = row[1]["strategy"]
        if read_details:
            df_new["column"] = row[1]["file_or_dir"]
        else:
            # column name contained in file names
            df_new["column"] = re.findall(col_pattern, row[1][file_col])[0][1]
        df_new["result_type"] = file_prefix
        dfs.append(df_new)
    return pd.concat(dfs, ignore_index=True)


def read_csv_files(df_experiment, read_details=True):
    """
    Reads data from the CSV files which were produced by an experiment, i.e. the results.

    Args:
        df_experiment: pd.DataFrame containing the conditions as well as names/path of the CSV files of an experiment.

    Returns:
        pd.DataFrame with all experiment conditions and (aggregated) scores
    """
    if read_details:
        result_types = [
            "impute_performance",
            "downstream_performance"
        ]
    else:
        result_types = [
            "impute_performance_std",
            "impute_performance_mean",
            "downstream_performance_std",
            "downstream_performance_mean"
        ]
    df_experiment = pd.concat(
        [_read_prefixed_csv_files(df_experiment, rt, read_details) for rt in result_types],
        ignore_index=True
    )
    df_experiment["missing_fraction"] = pd.to_numeric(df_experiment["missing_fraction"])

    ordered_columns = [
        "experiment", "imputer", "task", "missing_type", "missing_fraction", "strategy", "column",
        "result_type", "metric", "train", "test", "baseline", "corrupted", "imputed"
    ]
    assert len(ordered_columns) == df_experiment.shape[1]
    df_experiment = df_experiment[ordered_columns]

    return df_experiment
