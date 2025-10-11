from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class StochasticTournamentSelection(GaOperator):
    """
    Implements stochastic tournament selection.
    """

    def __init__(
        self, target_population: int, tournament_size: int = 3, p: float = 0.8
    ):
        """
        Initializes the StochasticTournamentSelection operator.

        :param target_population: The number of individuals to select.
        :param tournament_size: The number of individuals that compete in each tournament.
        :param p: Probability to select the best individual in the tournament. Must be in the range [0, 1].
        """

        self.target_population = target_population
        self.tournament_size = tournament_size
        self.p = p

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

        while len(selected_individuals) < self.target_population:
            indices = np.random.choice(
                len(args.population), size=self.tournament_size, replace=False
            )
            indiv = np.array(args.population)[indices]
            indiv = sorted(indiv, key=lambda obj: obj.value, reverse=maximize)
            threshold = self.p

            for crr in indiv:
                if np.random.rand() < threshold:
                    selected_individuals.append(crr)
                threshold = threshold * (1.0 - self.p)

        return selected_individuals
