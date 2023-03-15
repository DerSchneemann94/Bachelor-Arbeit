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
    def __init__(self, train_data, test_data, train_labels, test_labels, categorical_columns, numerical_columns, seed: Optional[int] = None):
        """
        Base class for task that get data from [OpenML](https://www.openml.org).

        Args:
            openml_id (int): ID of the to-be-fetched data from [OpenML](https://www.openml.org)
            train_size (float, optional): Defines the data split. Defaults to 0.8.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            is_image_data=False,
            seed=seed
        )


class ExternalDataMLRegressionTask(ExternalDataTask, RegressionTask):
    """
    Class that represents a regression task and gets data from [OpenML](https://www.openml.org).
    """
    pass


class ExternalDataMultiClassClassificationTask(ExternalDataTask, MultiClassClassificationTask):
    """
    Class that represents a multi-class classification task and gets data from [OpenML](https://www.openml.org).
    """
    pass


class ExternalDataBinaryClassificationTask(ExternalDataTask, BinaryClassificationTask):
    """
    Class that represents a binary classification task and gets data from [OpenML](https://www.openml.org).
    """
    pass
