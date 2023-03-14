class CategorialFeatureCharacteristics:

    def __init__(self, ordinal: bool, cardinality, column) -> None:
        self.ordnial = ordinal
        self.cardinality = cardinality
        self.column = column