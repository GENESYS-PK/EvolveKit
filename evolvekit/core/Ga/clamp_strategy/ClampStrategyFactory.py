from typing import Tuple, Callable

from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy


class ClampStrategyFactory:

    @staticmethod
    def get_clamp_strategy(
        strategy: GaClampStrategy,
    ) -> Callable[[float, Tuple[float, float]], float]:
        if strategy == GaClampStrategy.CLAMP:
            return ClampStrategyFactory.__strategy_clamp
        elif strategy == GaClampStrategy.BOUNCE:
            return ClampStrategyFactory.__strategy_bounce
        elif strategy == GaClampStrategy.OVERFLOW:
            return ClampStrategyFactory.__strategy_overflow
        elif strategy == GaClampStrategy.RANDOM:
            return ClampStrategyFactory.__strategy_random

    @staticmethod
    def __strategy_clamp(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    @staticmethod
    def __strategy_bounce(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    @staticmethod
    def __strategy_overflow(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()

    @staticmethod
    def __strategy_random(value: float, domain: Tuple[float, float]) -> float:
        raise NotImplementedError()
