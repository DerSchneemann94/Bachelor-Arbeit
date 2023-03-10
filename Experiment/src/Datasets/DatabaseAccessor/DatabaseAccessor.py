from Datasets.DatabaseAccessor.DatabaseAccesorDaos.OpenMl import OpenMlAccessor
from Datasets.DatabaseAccessor.DatabaseAccessorNotDefinedError import DatabaseAccessorNotDefinedError

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