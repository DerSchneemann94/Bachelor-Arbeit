import pandas as pd
from typing import Optional
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ..basis import (
    BinaryClassificationTask,
    MultiClassClassificationTask,
    RegressionTask,
    Task
)


class ExternalDataTask(Task):
    def __init__(self, data, labels, train_size: float = 0.8, seed: Optional[int] = 42):
        """
        Base class for task that get data from [OpenML](https://www.openml.org).

        Args:
            openml_id (int): ID of the to-be-fetched data from [OpenML](https://www.openml.org)
            train_size (float, optional): Defines the data split. Defaults to 0.8.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_size)

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            is_image_data=False,
            seed=seed
        )


class ExternalDataMLRegressionTask(ExternalDataTask, RegressionTask):
    pass


class ExternalDataMultiClassClassificationTask(ExternalDataTask, MultiClassClassificationTask):
    pass


class ExternalDataBinaryClassificationTask(ExternalDataTask, BinaryClassificationTask):
    pass
