from abc import ABC, abstractmethod


class DatabaseAccessorInterface(ABC):

    @staticmethod
    @abstractmethod
    def get_data_from_source(identifier:str):
        raise NotImplementedError