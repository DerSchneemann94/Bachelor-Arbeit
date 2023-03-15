import pandas as pd
from pathlib import Path

class PandaDataFrameCreator:

    @staticmethod
    def generate_dataframe_from_csv(csv_data):
        pass

    @staticmethod
    def generate_dataframe_from_path(path: Path):
        dataframe = pd.read_csv(path, skipinitialspace=True, index_col=False)
        return dataframe

    @staticmethod
    def split_dataframe_into_data_and_labels(dataframe: pd.DataFrame, target: str):
        label = pd.Series(dataframe[target])
        dataframe = dataframe.drop(target, axis=1)
        return dataframe, label
