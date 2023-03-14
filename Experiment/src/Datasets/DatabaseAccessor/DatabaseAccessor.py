from Datasets.DatabaseAccessor.DatabaseAccessorNotDefinedError import DatabaseAccessorNotDefinedError
from Datasets.DatabaseAccessor.Implementations.OpenMl.OpenMlAccessor import OpenMlAccessor

class DatabaseAccessor:

    database_accessor_dict = {
        "openml": OpenMlAccessor
    }

    @staticmethod
    def get_database_accessor(type: str):
        try:
            return DatabaseAccessor.database_accessor_dict[type]
        except:
            raise DatabaseAccessorNotDefinedError(type)