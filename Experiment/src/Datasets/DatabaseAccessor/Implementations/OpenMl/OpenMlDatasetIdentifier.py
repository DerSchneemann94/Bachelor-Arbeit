from Datasets.DatabaseAccessor.DatabaseIdentifier import DatabaseIdentifierInterface


class OpenmMlDatasetIdentifier(DatabaseIdentifierInterface):
    
    def __init__(self, openml_id):
        self.openml_id: str = openml_id 
        super().__init__()