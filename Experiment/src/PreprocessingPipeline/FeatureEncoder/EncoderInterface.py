from abc import ABC, abstractmethod

import pandas as pd

class EncoderInterface(ABC):

    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def transform_data(dataframe: pd.DataFrame, encoding_scheme):
        raise NotImplementedError

