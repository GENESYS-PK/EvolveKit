from enum import Enum, auto


class GaOpCategory(Enum):
    """
    Enum representing categories of genetic algorithm operators.

    This enum helps distinguish operator types from each other.

    :cvar SELECTION: Selection operator category.
    :cvar REAL_CROSSOVER: Real-valued crossover operator category.
    :cvar REAL_MUTATION: Real-valued mutation operator category.
    :cvar BIN_CROSSOVER: Binary crossover operator category.
    :cvar BIN_MUTATION: Binary mutation operator category.
    """

    SELECTION = auto()
    REAL_CROSSOVER = auto()
    REAL_MUTATION = auto()
    BIN_CROSSOVER = auto()
    BIN_MUTATION = auto()
