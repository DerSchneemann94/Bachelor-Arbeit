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

# First get all available datasets
all_datasets = openml.tasks.list_tasks(output_format="dataframe")
datasets = all_datasets.copy()
# Datasets without missing values
datasets = datasets[datasets["NumberOfInstancesWithMissingValues"] == 0]
# Active datasets
datasets = datasets[datasets["status"] == "active"]
# Rename 
datasets = datasets.rename(columns={"NumberOfSymbolicFeatures": "NumberOfCategoricalFeatures"})
# Only look at datasets with at least 5000 instances and at least 5 features
datasets = datasets[datasets["NumberOfInstances"] >= 3000]
datasets = datasets[datasets["NumberOfFeatures"] >= 5]
# Datasetws with max 100k instances and 25 features
datasets = datasets[datasets["NumberOfInstances"] <= 100000]
datasets = datasets[datasets["NumberOfFeatures"] <= 25]
# drop some corrupted datasets
datasets = datasets[~datasets["NumberOfClasses"].isna()]
# Can't work with sparse data
datasets["format"] = [_get_data_description_by_id(id, None)['format'] for id in datasets["did"]]
datasets = datasets[datasets["format"] != "Sparse_ARFF"]

# drop some unused columns
datasets = datasets.drop(columns=[
    "tid", "ttid", "task_type", "estimation_procedure", "evaluation_measures",
    "cost_matrix", "MaxNominalAttDistinctValues", "status", "target_value",
    "NumberOfMissingValues", "target_feature", "source_data", "number_samples",
    "source_data_labeled", "target_feature_event", "target_feature_left",
    "target_feature_right", "quality_measure", "NumberOfInstancesWithMissingValues", "format"
])
datasets = datasets.drop_duplicates()
datasets = datasets.drop_duplicates(["name"])


# ### Regression datasets
# 
# Regression datasets are datasets with `0` classes.
regression = datasets[datasets["NumberOfClasses"] == 0].copy()
regression = regression.drop_duplicates(["NumberOfFeatures", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"])
# Sort
regression["number_of_values"] = regression["NumberOfFeatures"] * regression["NumberOfInstances"]
regression = regression.sort_values(["number_of_values"])
# drop some unused columns
regression = regression.drop(columns=["MajorityClassSize", "MinorityClassSize", "NumberOfClasses", "number_of_values"])
# filter out duplicates by hand
#drop = [227, 42635, 1414, 42092]
#regression = regression[~regression["did"].isin(drop)]
regression = regression.reset_index(drop=True)
regression[:50]

# ### Classification datasets
# 
# #### Binary Classification datasets
# 
# Binary Classification datasets are datasets with `2` classes.
binary_classification = datasets[datasets["NumberOfClasses"] == 2].copy()
binary_classification = binary_classification.drop_duplicates(["MajorityClassSize", "MinorityClassSize", "NumberOfFeatures", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"])
# Sort
binary_classification["number_of_values"] = binary_classification["NumberOfFeatures"] * binary_classification["NumberOfInstances"]
binary_classification = binary_classification.sort_values(["number_of_values"])
# drop some unused columns
binary_classification = binary_classification.drop(columns=["NumberOfClasses", "number_of_values"])

# filter out duplicates by hand
# drop = [
#     41865, 41862, 41859, 41856, 41851, 41842, 41870, 41838,
#     41835, 41833, 41832, 41843, 41871, 41878, 41877, 41831,
#     41879, 41881, 41883, 41884, 41885, 41886, 41888, 41889,
#     41891, 41893, 41896, 41898, 41873, 41828, 41825, 41767,
#     41709, 41712, 41715, 41718, 41723, 41727, 41734, 41736,
#     41739, 41742, 41758, 41759, 41762, 41763, 41827, 41780,
#     41773, 41824, 41820, 41816, 41806, 41805, 41804, 41768,
#     41799, 41787, 41782, 41781, 41779, 41777, 41792, 821, 42178, 41860
# ]
# binary_classification = binary_classification[~binary_classification["did"].isin(drop)]
binary_classification = binary_classification.reset_index(drop=True)
binary_classification[:50]

# #### Multiclass Classification datasets
# 
# Multiclass Classification datasets are datasets with more than `2` classes.
multiclass_classification = datasets[datasets["NumberOfClasses"] > 2].copy()
multiclass_classification = multiclass_classification.drop_duplicates(["MajorityClassSize", "MinorityClassSize", "NumberOfFeatures", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"])
# Sort
multiclass_classification["number_of_values"] = multiclass_classification["NumberOfFeatures"] * multiclass_classification["NumberOfInstances"]
multiclass_classification = multiclass_classification.sort_values(["number_of_values"])
# drop some unused columns
multiclass_classification = multiclass_classification.drop(columns=["number_of_values"])
# filter out duplicates by hand
# drop = [119, 40678, 1222, 255]
# multiclass_classification = multiclass_classification[~multiclass_classification["did"].isin(drop)]
multiclass_classification = multiclass_classification.reset_index(drop=True)
multiclass_classification[:50]

assert (len(regression) + len(multiclass_classification) + len(binary_classification)) == 69

regression_for_paper = regression.copy()
regression_for_paper = regression_for_paper[["did", "name", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]]
regression_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]] = regression_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]].astype(int)
regression_for_paper = regression_for_paper.rename(columns={
    "did": "OpenML ID",
    "name": "Name",
    "NumberOfInstances": "# Instances",
    "NumberOfNumericFeatures": "# Num. Features",
    "NumberOfCategoricalFeatures": "# Cat. Features"
})
regression_table = regression_for_paper.to_latex(
    index=False,
    caption="Regression datasets.",
    label="tab:regression_data"
)
Path(get_project_root / "Datasets/datasets/regression_table.tex").write_text(regression_table)

binary_classification_for_paper = binary_classification.copy()
binary_classification_for_paper = binary_classification_for_paper[["did", "name", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]]
binary_classification_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]] = binary_classification_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]].astype(int)
binary_classification_for_paper = binary_classification_for_paper.rename(columns={
    "did": "OpenML ID",
    "name": "Name",
    "NumberOfInstances": "# Instances",
    "NumberOfNumericFeatures": "# Num. Features",
    "NumberOfCategoricalFeatures": "# Cat. Features"
})
bibinary_classification_table = binary_classification_for_paper.to_latex(
    index=False,
    caption="Binary classification datasets.",
    label="tab:binary_data"
)
Path(get_project_root / "Datasets/datasets/binary_table.tex").write_text(bibinary_classification_table)

multiclass_classification_for_paper = multiclass_classification.copy()
multiclass_classification_for_paper = multiclass_classification_for_paper[["did", "name", "NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]]
multiclass_classification_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]] = multiclass_classification_for_paper[["NumberOfInstances", "NumberOfNumericFeatures", "NumberOfCategoricalFeatures"]].astype(int)
multiclass_classification_for_paper = multiclass_classification_for_paper.rename(columns={
    "did": "OpenML ID",
    "name": "Name",
    "NumberOfInstances": "# Instances",
    "NumberOfNumericFeatures": "# Num. Features",
    "NumberOfCategoricalFeatures": "# Cat. Features"
})
multiclass_classification_table = multiclass_classification_for_paper.to_latex(
    index=False,
    caption="Multiclass classification datasets.",
    label="tab:multiclass_data"
)
Path(get_project_root / "Datasets/datasets/multiclass_table.tex").write_text(multiclass_classification_table)


