import os
import plotly.express as pl
import pandas as pd
import plotly.express as px
from Experiments.CategoricalFeatureEncodingExperiment import read_experiment
from pathlib import Path
from utils import get_project_root


EXPERIMENT_NAME = "2023-03-08_09:20"
EXPERIMENT_PATH = get_project_root() / "results" / EXPERIMENT_NAME
EXPERIMENT_PATH_BINARY = EXPERIMENT_PATH / "Binary-Classifikation"
EXPERIMENT_PATH_MULTI = EXPERIMENT_PATH / "Multi-Classifikation"
EXPERIMENT_PATH_REGRESSION = EXPERIMENT_PATH / "Regression-Classifikation"



experiments_binary = os.scandir(EXPERIMENT_PATH_BINARY)
csv_list = []
for experiment_directory in experiments_binary:
    path = Path(experiment_directory.path)
    mean_value_as_csv = list(path.rglob('*mean.csv'))
    csv_list.append(mean_value_as_csv)

pd_dataframe = pd.read_csv(csv_list[0][0])
pd_dataframe.drop(columns="Unnamed")
px.line(pd_dataframe)