from enum import Enum, auto


class GaOpCategory(Enum):
    """
    Enum representing categories of genetic algorithm operators.

    Members:
        SELECTION: Selection operator category.
        REAL_CROSSOVER: Real-valued crossover operator category.
        REAL_MUTATION: Real-valued mutation operator category.
        BIN_CROSSOVER: Binary crossover operator category.
        BIN_MUTATION: Binary mutation operator category.
    """

    SELECTION = auto()
    REAL_CROSSOVER = auto()
    REAL_MUTATION = auto()
    BIN_CROSSOVER = auto()
    BIN_MUTATION = auto()
