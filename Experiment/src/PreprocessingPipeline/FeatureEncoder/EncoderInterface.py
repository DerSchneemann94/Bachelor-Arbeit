from abc import ABC, abstractmethod

import pandas as pd

class EncoderInterface(ABC):

    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def transform_data_human_readable(dataframe: pd.DataFrame, encoding_scheme):
        raise NotImplementedError


    @abstractmethod
    def transform_data_compact(dataframe: pd.DataFrame, encoding_scheme):
        raise NotImplementedError