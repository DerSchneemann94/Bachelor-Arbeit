import json
import math
import random
import time
import pandas as pd
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional, Tuple
from jenga.corruptions.generic import MissingValues
from jenga.tasks.openml import OpenMLTask
from jenga.utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION
from numpy import nan
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator
from utils import get_project_root
class EvaluationError(Exception):
    """Exception raised for errors in Evaluation classes"""
    pass


class EvaluationResult(object):

    def __init__(self, task: OpenMLTask, categorical_feature_encoder_name: str):
        self._task = task
        self._finalized = False
        self.results: List[pd.DataFrame] = []
        self.downstream_performances: List[pd.DataFrame] = []
        self.elapsed_train_times: List[float] = []
        self.repetitions: int = 0
        self.categorical_feature_encoder_name: str = categorical_feature_encoder_name

        if self._task._task_type == BINARY_CLASSIFICATION or self._task._task_type == MULTI_CLASS_CLASSIFICATION:
            self._baseline_metric = ("F1_micro", "F1_macro", "F1_weighted")

        elif self._task._task_type == REGRESSION:
            self._baseline_metric = ("MAE", "MSE", "RMSE")

        self._baseline_performance = self._task.get_baseline_performance()
        

    def append(
        self,
        elapsed_time: float,
        model_categorial_feature_encoder: BaseEstimator
    ):
        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        predictions_with_desired_encoding = model_categorial_feature_encoder.predict(self._task.test_data)
        score_on_encoded = self._task.score_on_test_data(predictions_with_desired_encoding)

        dataframe = pd.DataFrame(
                {
                    "baseline: " + self._task.baseline_categorical_feature_encoder_name: {
                        self._baseline_metric[0]: self._baseline_performance[0],
                        self._baseline_metric[1]: self._baseline_performance[1],
                        self._baseline_metric[2]: self._baseline_performance[2]
                    },
                    "encoder: " + self.categorical_feature_encoder_name: {
                        self._baseline_metric[0]: score_on_encoded[0],
                        self._baseline_metric[1]: score_on_encoded[1],
                        self._baseline_metric[2]: score_on_encoded[2]
                    },
                }
            )

        self.downstream_performances.append(
            dataframe
        )
        self.elapsed_train_times.append(elapsed_time)
        self.repetitions += 1


    def finalize(self):
        if self._finalized:
            raise EvaluationError("Evaluation already finalized")
        results_mean_list = []
        results_std_list = []
        concatenated_results = pd.concat(self.downstream_performances)
        metrics = concatenated_results.index.unique()
        results_mean_list.append(
            pd.DataFrame(
                [
                    concatenated_results.loc[metric].mean() for metric in metrics
                ],
                index=metrics
            )
        )
        results_std_list.append(
            pd.DataFrame(
                [
                    concatenated_results.loc[metric].std() for metric in metrics
                ],
                index=metrics
            )
        )
        self.downstream_performance = results_mean_list[0]
        self.downstream_performance_std = results_std_list[0]
        self.elapsed_train_time = mean(self.elapsed_train_times)
        self.elapsed_train_time_std = stdev(self.elapsed_train_times)
        self._finalized = True
        return self


class Evaluator(object):
    def __init__(
        self,
        task: OpenMLTask,
        categorical_feature_encoder_name: str,
        categorical_feature_encoder: BaseEstimator,
        numerical_feature_encoder_name: str,
        numerical_feature_encoder: BaseEstimator,
        path: Optional[Path] = None,
    ):
        self._task = task
        self._result: Optional[Dict[str, EvaluationResult]] = None
        self._categorical_feature_encoder_name = categorical_feature_encoder_name
        self._categorical_feature_encoder = categorical_feature_encoder
        self._numerical_feature_encoder_name = numerical_feature_encoder_name
        self._numerical_feature_encoder = numerical_feature_encoder
        self._path = path 
        # fit task's baseline model and get performance
        self._task.fit_baseline_model()
        

    @staticmethod
    def report_results(result_dictionary: Dict[str, EvaluationResult]) -> None:
        target_columns = list(result_dictionary.keys())

        print(f"Evaluation result contains {len(target_columns)} target columns: {', '.join(target_columns)}")
        print("All are in a round-robin fashion imputed and performances are as follows:\n")

        for key, value in result_dictionary.items():
            print(f"Target Column: {key} - Necessary train time in seconds: {round(value.elapsed_train_time, 4)}")
            print(value.result)
            print()
            print(value.downstream_performance)
            print("\n")

    def evaluate(self, number_of_repetitions):
        result = None 
        result_temp = EvaluationResult(self._task, self._categorical_feature_encoder_name)
        feature_transformer = self._task.create_data_preprocessor(
            self._categorical_feature_encoder_name,
            self._categorical_feature_encoder,
            self._numerical_feature_encoder_name,
            self._numerical_feature_encoder
            )        
        start_time = time.time()
        model_categorial_feature_encoder = self._task.fit_model_feature_transformer(feature_transformer, self._task.train_data, self._task.test_labels)            
        elapsed_time = time.time() - start_time
        for _ in range(number_of_repetitions):
            # NOTE: masks are DataFrames => append expects Series
            result_temp.append(
                elapsed_time=elapsed_time,
                model_categorial_feature_encoder=model_categorial_feature_encoder
            )
        result = result_temp.finalize()
        self._result = result
        self._save_results()
        return self

    def report(self) -> None:
        if self._result is None:
            raise EvaluationError("Not evaluated yet. Call 'evaluate' first!")
        else:
            self.report_results(self._result)

    def _discard_values(
        self,
        task: OpenMLTask,
        to_discard_columns: List[str],
        missing_fraction: float,
        missing_type: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        columns = to_discard_columns
        fraction = missing_fraction / len(columns)
        missing_values = []
        for column in columns:
            missing_values.append(MissingValues(column=column, fraction=fraction, missingness=missing_type))
        # Apply all
        train_data = task.train_data.copy()
        test_data = task.test_data.copy()
        for missing_value in missing_values:
            train_data = missing_value.transform(train_data)
            test_data = missing_value.transform(test_data)
        return (train_data, test_data)

    def _save_results(self):
        if self._path is not None:
            #print("Path:  ", self._path)
            self._path.mkdir(parents=True, exist_ok=True)
            #Mean results
            self._result.downstream_performance.to_csv(self._path / f"downstream_performance_mean.csv")
            # Standard deviation results
            self._result.downstream_performance_std.to_csv(self._path / f"downstream_performance_std.csv")
            Path(self._path / f"elapsed_train_time.json").write_text(json.dumps(
                {
                    "mean": self._result.elapsed_train_time,
                    "std": self._result.elapsed_train_time_std
                }
            ))
            results_path = self._path / "report for every experiment"
            results_path.mkdir(parents=True, exist_ok=True)    
            for index, (performance_data_frame, elapsed_train_time) in enumerate(
                zip(
                    self._result.downstream_performances,
                    self._result.elapsed_train_times                    
                )
            ):
                performance_data_frame.to_csv(results_path / f"downstream_performance_rep_{index}.csv")
                Path(results_path / f"elapsed_train_time_rep_{index}.json").write_text(json.dumps(elapsed_train_time))


class SingleColumnEvaluator(Evaluator):
    """
    Evaluate Missing Value effects on single column.
    """

    def __init__(
        self,
        task: OpenMLTask,
        categorical_feature_encoder_name: str,
        categorical_feature_encoder: BaseEstimator,
        numerical_feature_encoder_name: str,
        numerical_feature_encoder: BaseEstimator,
        path: Optional[Path] = None,
        
    ):

        super().__init__(
            task=task,
            path=path,
            categorical_feature_encoder=categorical_feature_encoder,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder=numerical_feature_encoder,
            numerical_feature_encoder_name=numerical_feature_encoder_name
        )


class MultipleColumnsEvaluator(Evaluator):
    """
    Evaluate Missing Value effects on multiple columns.
    """

    def __init__(
        self,
        task: OpenMLTask,
        categorical_feature_encoder_name: str,
        categorical_feature_encoder: BaseEstimator,
        numerical_feature_encoder_name: str,
        numerical_feature_encoder: BaseEstimator,
        path: Optional[Path] = None,
    ):

        super().__init__(
            task=task,
            path=path,
            categorical_feature_encoder=categorical_feature_encoder,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder=numerical_feature_encoder,
            numerical_feature_encoder_name=numerical_feature_encoder_name
        )


class SingleColumnAllMissingEvaluator(Evaluator):
    """
    Evaluate Missing Value effects on single column when all columns contain missing values.
    """

    def __init__(
        self,
        task: OpenMLTask,
        categorical_feature_encoder_name: str,
        categorical_feature_encoder: BaseEstimator,
        numerical_feature_encoder_name: str,
        numerical_feature_encoder: BaseEstimator,
        path: Optional[Path] = None,
    ):

        super().__init__(
            task=task,
            path=path,
            categorical_feature_encoder=categorical_feature_encoder,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder=numerical_feature_encoder,
            numerical_feature_encoder_name=numerical_feature_encoder_name
        )


class MultipleColumnsAllMissingEvaluator(Evaluator):
    """
    Evaluate Missing Value effects on multiple columns when all columns contain missing values.
    """

    def __init__(
        self,
        task: OpenMLTask,
        categorical_feature_encoder_name: str,
        categorical_feature_encoder: BaseEstimator,
        numerical_feature_encoder_name: str,
        numerical_feature_encoder: BaseEstimator,
        path: Optional[Path] = None,
    ):

        super().__init__(
            task=task,
            path=path,
            categorical_feature_encoder=categorical_feature_encoder,
            categorical_feature_encoder_name=categorical_feature_encoder_name,
            numerical_feature_encoder=numerical_feature_encoder,
            numerical_feature_encoder_name=numerical_feature_encoder_name
        )
