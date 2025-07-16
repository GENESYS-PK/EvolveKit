from typing import Tuple, Callable
import numpy as np

from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy


def get_clamp_strategy(
    strategy: GaClampStrategy,
) -> Callable[[float, Tuple[float, float]], float]:

    def strategy_clamp(value: float, domain: Tuple[float, float]) -> float:
        lower, upper = domain
        return max(lower, min(value, upper))

    def strategy_bounce(value: float, domain: Tuple[float, float]) -> float:
        lower, upper = domain
        if value > upper:
            value = upper - (value - upper)
        elif value < lower:
            value = lower + (lower - value)
        return value

    def strategy_overflow(value: float, domain: Tuple[float, float]) -> float:
        lower, upper = domain
        if value > upper:
            value = lower + (value - upper)
        elif value < lower:
            value = upper - (lower - value)
        return value

    def strategy_random(value: float, domain: Tuple[float, float]) -> float:
        lower, upper = domain
        if lower <= value <= upper:
            return value
        return np.random.uniform(lower, upper)

    MAP_STRATEGY_NAME_TO_FUNCTION = {
        GaClampStrategy.CLAMP: strategy_clamp,
        GaClampStrategy.BOUNCE: strategy_bounce,
        GaClampStrategy.OVERFLOW: strategy_overflow,
        GaClampStrategy.RANDOM: strategy_random,
    }

    return MAP_STRATEGY_NAME_TO_FUNCTION[strategy]
