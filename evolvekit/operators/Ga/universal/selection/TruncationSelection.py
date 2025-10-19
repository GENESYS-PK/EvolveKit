from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class TruncationSelection(GaOperator):
    """
    Implements truncation selection.
    """

    def __init__(self, target_population: int):
        """
        Initializes the TruncationSelection operator.

        :param target_population: The number of individuals to select.
        """

        self.target_population = target_population

    def category(self) -> GaOpCategory:
        """
        Returns the category of this operator.

        :returns: GaOpCategory, SELECTION category of the operator.
        """
        return GaOpCategory.SELECTION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs selection.

        :param args: GaOperatorArgs, includes population and evaluator.
        :returns: list of selected individuals.
        """

        if len(args.population) <= self.target_population:
            raise RuntimeError(
                "Target population must be smaller than the whole population"
            )

        maximize = args.evaluator.extremum() == GaExtremum.MAXIMUM
        sorted_pop = sorted(
            args.population, key=lambda obj: obj.value, reverse=maximize
        )

        return sorted_pop[: self.target_population]
