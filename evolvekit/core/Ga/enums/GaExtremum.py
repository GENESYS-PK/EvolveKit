from enum import Enum, auto


class GaExtremum(Enum):
    """
    Enum represents the optimization criterion for the genetic algorithm.

    This enum helps determine if larger values returned by the fitness
    function are better or worse than the smaller ones.

    :cvar MINIMUM: Search for the minimum value.
    :cvar MAXIMUM: Search for the maximum value.
    """

    MINIMUM = auto()
    MAXIMUM = auto()
