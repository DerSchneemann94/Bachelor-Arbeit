#!/usr/bin/env python
# coding: utf-8

# Basically, there are 3 different types of tasks:
# 
# 1. Regression
# 2. Binary Classification
# 3. Mulitclass Classification
# 
# However, we can split them further:
# 
# 3. Mulitclass Classification into ..
#     - 3-class
#     - 4-class
#     - ...

import openml
import json
import pandas as pd
from sklearn.datasets._openml import _get_data_description_by_id
from pathlib import Path
from utils import get_project_root

# ## Filtering available datasets
# 
# - Active datasets
# - No missing values
#     - If we have missing values in the original dataset, we can not trust the downstream task performance changes
# - 3k to 100k instances
# - 5 to 25 features
# - Drop duplicated datasets
# - Drop datasets with the same name
# - Drop datasets with no information about the number of classes
# - Drop datasets where number of features, number of instances, class distribution are the same (high probability to be duplicated)
# - Remove some duplicated by hand
# - Make sure the dataset holds categorical values
# - (At the end) we only use 50 datasets for each task (regression, binary, multiclass)
project_root = get_project_root()
path_to_results = project_root / "src/Data/Datasets_identify/openml/openml_statistics_from_all_openml_datasets"
path_to_corrupted = path_to_results / "corrupted_datasets.txt"
unsuccessfull_datasets = []

# First get all available datasets
all_datasets = openml.tasks.list_tasks(output_format="dataframe")
datasets = all_datasets.copy()
# Datasets without missing values
# datasets = datasets[datasets["NumberOfInstancesWithMissingValues"] == 0]
# Active datasets
datasets = datasets[datasets["status"] == "active"]
# Rename 
datasets = datasets.rename(columns={"NumberOfSymbolicFeatures": "NumberOfCategoricalFeatures"})
# Only look at datasets with at least 5000 instances and at least 5 features
datasets = datasets[datasets["NumberOfInstances"] >= 1000]
datasets = datasets[datasets["NumberOfFeatures"] >= 3]
# drop some corrupted datasets
datasets = datasets[~datasets["NumberOfClasses"].isna()]

tasks = {
    "Regression": datasets[datasets["NumberOfClasses"] == 0].copy(),
    "Binary-Classification": datasets[datasets["NumberOfClasses"] == 2].copy(),
    "Multiple-Classification": datasets[datasets["NumberOfClasses"] >= 2].copy()
}

dataset_investigator = OrdinalFeatureInvestigator()
statistics_collection = {}
number_of_datasets = datasets.index.size
number_of_dataset = 1
for task_type in tasks.keys():
    datasets = tasks[task_type]
    path = path_to_results / task_type
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    for index  in range(datasets.index.size):
        print("Datasetnr.:  " + str(number_of_dataset) + " /  " + str(number_of_datasets))
        number_of_dataset += 1
        dataset = datasets.iloc[index]
        openml_id = dataset["did"]
        try:
            statistics = dataset_investigator.investigate_dataset(openml_id, dataset)
            statistics_collection[openml_id] = statistics
            if statistics is not None:
                with open(path / (openml_id + ".txt"), "w") as file:
                    json.dump(statistics, file, indent=0)
        except:
            unsuccessfull_datasets.append(openml_id)
            continue

path_to_corrupted.write_text(str(unsuccessfull_datasets))
print("check!")