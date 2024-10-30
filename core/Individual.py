from dataclasses import dataclass
from typing import Any


@dataclass
class Individual:
    """
    A class that stores data about an individual.

    :param chromosome: A chromosome representing the individual (can be of any type).
    :param value: The fitness value for this individual, expressed as a floating point number.
    """

    chromosome: Any
    value: float
