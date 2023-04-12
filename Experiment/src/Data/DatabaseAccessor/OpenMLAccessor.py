from Data.DatabaseAccessor.DatabaseAccessorInterface import DatabaseAccessorInterface
from sklearn.datasets import fetch_openml


class OpenMLAccessor(DatabaseAccessorInterface):


    @staticmethod
    def get_data_from_source(identifier: str):
        try:
            dataframe = fetch_openml(data_id=identifier, as_frame=True, return_X_y=False, cache=False)
            return dataframe
        except Exception as error:
            print("fetching database with openml_id: " + identifier + " has not been successful.")
            raise error

