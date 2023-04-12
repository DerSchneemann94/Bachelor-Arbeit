from typing import Dict

from PreprocessingPipeline.FeatureEncoder.EncoderInterface import EncoderInterface


class PipelineInitializer:
    
    def __init__(self) -> None:
        self._class_mapping = {

        }


    @staticmethod
    def intialize_pipeline() -> Dict[str, EncoderInterface]:
        pass
