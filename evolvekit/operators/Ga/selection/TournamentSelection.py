from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class TournamentSelection(GaOperator):
    """
    Implements tournament selection.
    """

    def __init__(self, target_population: int, tournament_size: int = 3):
        """
        Initializes the TournamentSelection operator.

        :param target_population: The number of individuals to select.
        :param tournament_size: The number of individuals that compete in each tournament.
        """

        self.target_population = target_population
        self.tournament_size = tournament_size

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
        pop_size = len(args.population)
        maximize = args.evaluator.extremum() == GaExtremum.MAXIMUM

        while len(selected_individuals) < self.target_population:
            bestValue = float("-inf") if maximize else float("inf")
            bestIdx = 0

            for _ in range(self.tournament_size):
                idx = np.random.randint(0, pop_size)
                value = args.population[idx].value

                if maximize == (value > bestValue):
                    bestValue = value
                    bestIdx = idx

            selected_individuals.append(args.population[bestIdx])

        return selected_individuals
