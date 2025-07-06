from abc import ABC, abstractmethod
from typing import List
from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class GaOperator(ABC):

    def initialize(self, state: GaState):
        pass

    @abstractmethod
    def category(self) -> GaOpCategory:
        pass

    @abstractmethod
    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        pass
