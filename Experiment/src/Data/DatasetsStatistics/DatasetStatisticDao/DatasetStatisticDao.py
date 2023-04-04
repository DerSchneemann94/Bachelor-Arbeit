from abc import ABC, abstractmethod


class DatasetStatisticDao(ABC):

    @staticmethod
    @abstractmethod
    def write_json_to_file(path, serializable_file):
        raise NotImplementedError
    

    @staticmethod
    @abstractmethod
    def read_from_json(path, serializable_file):
        raise NotImplementedError
    