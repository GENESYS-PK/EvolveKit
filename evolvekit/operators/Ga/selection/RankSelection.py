from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class RankSelection(GaOperator):
    def __init__(self, target_population: int):
        """
        Initializes the RankSelection operator.

        :param target_population: int, number of individuals to select
        :returns: None
        :raises ValueError: if target_population is not an integer
        """
        self.target_population = target_population

    def category(self) -> GaOpCategory:
        """
        Returns the category of this operator.

        :returns: GaOpCategory, SELECTION category of the operator
        :raises ValueError: never raises
        """
        return GaOpCategory.SELECTION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs the rank-based selection on the population.

        For more details about stochastic acceptance roulette-wheel,
        see: https://arxiv.org/abs/1109.3627

        :param args: GaOperatorArgs, includes population and evaluator
        :returns: list of selected GaIndividual based on their rank
        :raises ValueError: if args is not a GaOperatorArgs instance
        """
        selected_individuals = []
        pop_size = len(args.population)
        sorted_pop = sorted(
            args.population,
            key=lambda obj: obj.value,
            reverse=(args.evaluator.extremum == GaExtremum.MINIMUM),
        )

        for i in range(pop_size):
            sorted_pop[i].value = i + 1

        while len(selected_individuals) < self.target_population:
            index = np.random.randint(0, pop_size)
            individual = sorted_pop[index]
            weight = (index + 1) / pop_size

            if np.random.rand() < weight:
                selected_individuals.append(individual)

        return selected_individuals
