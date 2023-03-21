import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    if seed:
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


REGRESSION = 0
BINARY_CLASSIFICATION = 1
MULTI_CLASS_CLASSIFICATION = 2

AVAILABLE_TASK_TYPES = [
    REGRESSION,
    BINARY_CLASSIFICATION,
    MULTI_CLASS_CLASSIFICATION
]
