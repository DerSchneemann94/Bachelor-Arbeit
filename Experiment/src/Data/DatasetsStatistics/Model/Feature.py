from abc import ABC, abstractmethod


class Feature(ABC):
    def __init__(self, type) -> None:
        self.type: str = type


    @abstractmethod
    def get_feature_contents():
        raise NotImplementedError
    

class CategoricalFeature(Feature):
    def __init__(self, ordinal, cardinality) -> None:
        super().__init__(type="categorical")
        self.ordinal: bool = ordinal
        self.cardinality: int = cardinality
    

    def get_feature_contents(self):
        return {
            "type": self.type,
            "ordinal": self.ordinal,
            "cardinality": self.cardinality 
        }
    

class NumericFeature(Feature):
    def __init__(self) -> None:
        super().__init__(type="numeric")


    def get_feature_contents(self):
        return {
            "type": self.type
        }        