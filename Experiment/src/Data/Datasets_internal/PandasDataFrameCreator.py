import pandas as pd
from pathlib import Path

class PandasDataFrameCreator:

    @staticmethod
    def generate_dataframe_from_path(path:Path):
        try:
            dataframe = pd.read_csv(path)
            return dataframe
        except Exception as error:
            raise error
        

    @staticmethod
    def split_dataframe_into_data_and_labels(dataframe: pd.DataFrame, target: str):
        label = pd.Series(dataframe[target])
        dataframe = dataframe.drop(target, axis=1)
        return dataframe, label
