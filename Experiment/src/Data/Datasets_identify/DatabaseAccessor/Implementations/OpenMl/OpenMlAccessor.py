from sklearn.datasets import fetch_openml
from typing import List
from Datasets.DatabaseAccessor import DatabaseAccessorInterface, DatabaseIdentifier
from Datasets.DatabaseAccessor.DatabaseAccesorDaos.OpenMl import OpenMlDatasetIdentifier
class OpenMlAccessor(DatabaseAccessorInterface):

    def get_data_set(identifier: OpenMlDatasetIdentifier):
        dataset = fetch_openml(identifier.openml_id)
        
