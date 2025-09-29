from enum import Enum, auto


class GaExtremum(Enum):
    """
    Enum representing the optimization criterion for the genetic algorithm.

    Members:
        MINIMUM: Search for the minimum value.
        MAXIMUM: Search for the maximum value.
    """

    MINIMUM = auto()
    MAXIMUM = auto()
