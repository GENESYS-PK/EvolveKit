from enum import Enum, auto


class GaClampStrategy(Enum):
    CLAMP = auto()
    BOUNCE = auto()
    OVERFLOW = auto()
    RANDOM = auto()
