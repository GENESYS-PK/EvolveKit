from enum import Enum, auto


class GaClampStrategy(Enum):
    NONE = auto()
    CLAMP = auto()
    BOUNCE = auto()
    OVERFLOW = auto()
    RANDOM = auto()
