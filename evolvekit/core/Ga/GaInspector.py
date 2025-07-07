from abc import ABC, abstractmethod

from evolvekit.core.Ga.enums.GaAction import GaAction
from evolvekit.core.Ga.GaStatistics import GaStatistics


class GaInspector(ABC):

    def initialize(self):
        pass

    @abstractmethod
    def inspect(self, stats: GaStatistics) -> GaAction:
        pass

    def finish(self, stats: GaStatistics):
        pass
