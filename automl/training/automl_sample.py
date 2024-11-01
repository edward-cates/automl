from abc import ABC, abstractmethod

class AutomlSample(ABC):
    """
    Samples need to perform certain functions (like being prepared for model).
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_model_input(self):
        pass
