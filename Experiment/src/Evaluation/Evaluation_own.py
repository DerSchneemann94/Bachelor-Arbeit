import pandas as pd
from jenga.utils import set_seed
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional
from jenga.tasks.ExternalDataTask import ExternalDataTask
from jenga.utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION
from numpy import nan
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator
from utils import get_project_root


class EvaluationError(Exception):
    """Exception raised for errors in Evaluation classes"""
    pass


class Evaluationresult_own(object):
    def __init__(self, task: ExternalDataTask):
        self._task = task
        self._finalized = False
        self.results: List[pd.DataFrame] = []
        self.downstream_performances: List[pd.DataFrame] = []
        self.elapsed_train_times: List[float] = []
        self.repetitions: int = 0

        if self._task._task_type == BINARY_CLASSIFICATION or self._task._task_type == MULTI_CLASS_CLASSIFICATION:
            self._baseline_metric = ("F1_micro", "F1_macro", "F1_weighted")

        elif self._task._task_type == REGRESSION:
            self._baseline_metric = ("MAE", "MSE", "RMSE")

        

    def append(
        self,
        model: BaseEstimator
    ):
        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        predictions_with_desired_encoding = model.predict(self._task.test_data)
        score_on_encoded = self._task.score_on_test_data(predictions_with_desired_encoding)

        dataframe = pd.DataFrame(
                {   "performance":
                        {
                        self._baseline_metric[0]: score_on_encoded[0],
                        self._baseline_metric[1]: score_on_encoded[1],
                        self._baseline_metric[2]: score_on_encoded[2]
                        }
                }
            )

        self.downstream_performances.append(
            dataframe
        )



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
                    concatenated_results.loc[metric] for metric in metrics
                ],
                index=metrics
            )
        )
        results_std_list.append(
            pd.DataFrame(
                [
                    concatenated_results.loc[metric] for metric in metrics
                ],
                index=metrics
            )
        )
        self.downstream_performance = results_mean_list[0]
        self._finalized = True
        return self.downstream_performance


class Evaluator_own(object):
    def __init__(
        self,
        task: ExternalDataTask,
        path: Optional[Path] = None,
        seed: int = 42,
        
    ):
        self._task = task
        self._result: Optional[Dict[str, Evaluationresult_own]] = None
        self._path = path
        self._seed = seed

        set_seed(self._seed)


    def evaluate(self):
        result = None 
        result_temp = Evaluationresult_own(self._task)
        model = self._task.fit_model(self._task.train_data, self._task.test_labels)            
        # NOTE: masks are DataFrames => append expects Series
        result_temp.append(
            model=model
        )
        result = result_temp.finalize()
        self._result = result
        return result
