from enum import StrEnum


class GaClampStrategy(StrEnum):
    CLAMP = "CLAMP"
    BOUNCE = "BOUNCE"
    OVERFLOW = "OVERFLOW"
    RANDOM = "RANDOM"
