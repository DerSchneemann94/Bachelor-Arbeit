from typing import List


class CategoricalFeatureCharacteristics:

    def __init__(self, ordinal: bool, cardinality: int, name: str) -> None:
        self.ordinal = ordinal
        self.cardinality = cardinality
        self.name = name



class DatasetCharacteristics:
    def __init__(self, instances, features, categorical_characteristics:List[CategoricalFeatureCharacteristics]) -> None:
        self.categorical_feature_characteristics = categorical_characteristics
        self.instances = instances
        self.features = features