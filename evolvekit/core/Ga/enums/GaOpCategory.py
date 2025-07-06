from enum import Enum, auto


class GaOpCategory(Enum):
    SELECTION = auto()
    REAL_CROSSOVER = auto()
    REAL_MUTATION = auto()
    BIN_CROSSOVER = auto()
    BIN_MUTATION = auto()
