class DataBaseProviderNotFroundError(Exception):

    def __init__(self, *args: object, type: str) -> None:
        super().__init__(*args)
        self.type = type