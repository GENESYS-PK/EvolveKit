from abc import ABC, abstractmethod

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from typing import List, Tuple


class GaEvaluator(ABC):

    @abstractmethod
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        pass

    @abstractmethod
    def extremum(self) -> GaExtremum:
        pass

    def real_domain(self) -> List[Tuple[float, float]]:
        return []

    def bin_length(self) -> int:
        return 0
