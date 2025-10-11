from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class UnbiasedTournamentSelection(GaOperator):
    """
    Implements unbiased tournament selection.

    For more details see:
    https://www.cs.colostate.edu/~genitor/2005/GECCO247.pdf
    """

    def __init__(self):
        pass

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

        selected_individuals = []
        maximize = args.evaluator.extremum() == GaExtremum.MAXIMUM
        shuffled_pop = np.random.permutation(args.population)

        for a, b in zip(args.population, shuffled_pop):
            selected_individuals.append(a if maximize == (a.value > b.value) else b)

        return selected_individuals
