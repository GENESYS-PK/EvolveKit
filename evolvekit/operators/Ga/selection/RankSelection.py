from typing import List
import numpy as np

from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class RankSelection(GaOperator):
    def __init__(self, target_population: int):
        """Initializes the RankSelection operator.

        Args:
            target_population (int): The number of individuals to select
            for the target population.
        """
        self.target_population = target_population

    def initialize(self, state: GaState):
        """Initializes the state for the RankSelection operator.

        Args:
            state (GaState): The current state of the genetic algorithm,
            containing population and other parameters.
        """
        self.state = state

    def category(self) -> GaOpCategory:
        """Returns the category of this operator.

        Returns:
            GaOpCategory: The category of the operator, which is SELECTION.
        """
        return GaOpCategory.SELECTION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs the rank-based selection on the population.

        For more details about stochastic acceptance roulette-wheel, see:
        https://arxiv.org/abs/1109.3627

        Args:
            args (GaOperatorArgs): The arguments for the operator, including
            the population and evaluator.

        Returns:
            List[GaIndividual]: A list of selected individuals based on
            their rank.
        """
        selected_individuals = []
        pop_size = self.state.population_size
        sorted_pop = sorted(
            args.population,
            key=lambda obj: obj.value,
            reverse=(args.evaluator.extremum == GaExtremum.MAXIMUM),
        )

        for i in range(pop_size):
            sorted_pop[i].value = i + 1

        while len(selected_individuals) < self.target_population:
            index = np.random.randint(0, pop_size)
            individual = args.population[index]
            weight = (index + 1) / pop_size

            if np.random.rand() < weight:
                selected_individuals.append(individual)

        return selected_individuals
