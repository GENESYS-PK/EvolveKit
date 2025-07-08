from typing import Tuple, Callable

from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy


def get_clamp_strategy(
    strategy: GaClampStrategy,
) -> Callable[[float, Tuple[float, float]], float]:
    def strategy_clamp(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    def strategy_bounce(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    def strategy_overflow(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    def strategy_random(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    MAP_STRATEGY_NAME_TO_FUNCTION = {
        GaClampStrategy.CLAMP: strategy_clamp,
        GaClampStrategy.BOUNCE: strategy_bounce,
        GaClampStrategy.OVERFLOW: strategy_overflow,
        GaClampStrategy.RANDOM: strategy_random,
    }

    return MAP_STRATEGY_NAME_TO_FUNCTION[strategy]
