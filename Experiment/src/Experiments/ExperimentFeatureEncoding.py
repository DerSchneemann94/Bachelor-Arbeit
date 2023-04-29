import logging
import traceback
import pandas as pd
from jenga.tasks.ExternalDataTask import ExternalDataTask
from Evaluation.Evaluation_own import Evaluator_own
from jenga.utils import set_seed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from utils import get_project_root


logger = logging.getLogger()

class ExperimentFeatureEncoding(object):
    def __init__(
        self,
        encoded_data: pd.DataFrame,
        original_data: pd.DataFrame,
        dataset_characteristic: any,
        labels: pd.Series,
        task_id_class_tuples: List[Tuple[int, Callable[..., ExternalDataTask]]],
        elapsed_time: float,
        model_name: str,
        experiment_configuration: Dict,
        base_path: str = 'results',
        experiment_name: Optional[str] = None,
        seed: int = 42
    ):
        self._encoded_data = encoded_data
        self._original_data = original_data
        self._dataset_characteristic = dataset_characteristic
        self._labels = labels
        self._task_id_class_tuples = task_id_class_tuples
        self._experiment_name = experiment_name
        self._result: Dict[int, Dict[str, Dict[float, Dict[str, Evaluator_own]]]] = dict()
        self._seed = seed
        self._elapsed_time = elapsed_time
        self._experiment_configuration = experiment_configuration
        self._task = None
        self._model_name = model_name

        project_root = get_project_root()
        self._base_path = project_root / base_path

        if self._experiment_name is None:
            self._experiment_name = datetime.now().strftime("%Y-%m-%d_%H:%M")

        if self._seed:
            set_seed(self._seed)

        

    def run(self):
        for task_id, task_class in self._task_id_class_tuples:
            self._result[task_id] = {}
            self.task:ExternalDataTask = task_class(seed=self._seed, data=self._encoded_data, labels=self._labels, model_name=self._model_name)
            try:
                evaluator = Evaluator_own(
                    task=self.task,
                    seed=self._seed,
                )
                evalutation_result, model_hyperparameter = evaluator.evaluate()
                result = {
                    "performance" : evalutation_result,
                    "elapsed_train_time": self._elapsed_time,
                    "experiment_config": self._experiment_configuration,
                    "dataset_characteristic": self._dataset_characteristic,
                    "test_data": self.task.test_data,
                    "train_data": self.task.train_data,
                    "model_hyperparameter": model_hyperparameter
                }
                return result
                
                
            except Exception as error:
                error = traceback.format_exc()
                self._base_path.mkdir(parents=True, exist_ok=True)
                Path(self._base_path / "error.txt").write_text(str(error))
                result = error
                self._result[task_id] = result
                raise error


        logger.info(f"Experiment Finished! - Results are at: {self._base_path.parent}")






