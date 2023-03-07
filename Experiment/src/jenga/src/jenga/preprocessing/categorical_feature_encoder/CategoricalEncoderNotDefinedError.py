class CategoricalEncoderNotDefinedError(Exception):
    
    def __init__(self, encoder_name) -> None:
        self.encoder_name = encoder_name