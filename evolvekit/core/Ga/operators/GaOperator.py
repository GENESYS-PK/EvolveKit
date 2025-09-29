from abc import ABC, abstractmethod
from typing import List

from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class GaOperator(ABC):
    """
    Abstract base class for genetic algorithm operators.
    """

    def initialize(self, state: GaState):
        """
        Optional method that can be overridden in the operator implementation. It will be automatically
        called before the evolutionary loop. Sometimes it may be useful if some exotic operator
        requires additional configuration before starting its operation.

        :param state: The current state of the genetic algorithm.
        :type state: GaState
        """
        pass

    @abstractmethod
    def category(self) -> GaOpCategory:
        """
        Each operator must implement this method.
        Its purpose is only to return the category to which the given operator belongs.
        Thanks to this, when the library core receives an object of the GaOperator class,
        it will know what is hidden under this object (example: binary crossover).

        :returns: The operator category.
        :rtype: GaOpCategory
        """
        pass

    @abstractmethod
    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        This is the method where the algorithm for a given operator will be implemented. The method takes
        a special `GaOperatorArgs` object, which contains the data the operator may require, and as a result,
        this method must return a population. In the case of selection, it will be the selected population; in the case
        of crossover – a generation of offspring; and in the case of mutation – a generation of mutated offspring.

        :param args: Arguments required for the operator's execution.
        :type args: GaOperatorArgs
        :returns: List of individuals resulting from the operation.
        :rtype: List[GaIndividual]
        """
        pass
