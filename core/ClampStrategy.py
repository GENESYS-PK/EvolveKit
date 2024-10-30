import numpy as np
from abc import ABC, abstractmethod
from Population import Population

class ClampStrategy(ABC):
    """
    An abstract base class, serving as a blueprint for clamp startegy implementations
    """
    @abstractmethod
    def clamp(variable_domains: List[Tuple[float, float]], population: Population) -> None:
        """
        Force every individual in population into specified domain. 
        This method is performed in place.

        :param variable_domains: A list of tuples of two floats, specifying 
        upper and lower bounds of values in chromosome.
        :param population: A population for which individuals are to be 
        clamped.
        """
        pass