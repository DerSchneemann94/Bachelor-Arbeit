from abc import ABC, abstractmethod
from typing import List

from Datasets.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface

#base abstration for data base interation; in case future databases do not provide a convenient way of accessing its contents
class DatabaseAccessorInterface(ABC): 
    
    @abstractmethod
    def get_data_set(data_set_identifier: DatabaseIdentifierInterface):
        raise NotImplementedError


