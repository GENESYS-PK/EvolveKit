from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class SaRouletteWindowSelection(GaOperator):
    """
    Stochastic acceptance roulette-wheel selection with windowing.

    For more details about stochastic acceptance roulette-wheel see:
    https://arxiv.org/abs/1109.3627

    For more details about windowing see:
    https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Practical%20Handbook%20of%20GENETIC%20ALGORITHMS%2C%20Volume%20II/ganf3.pdf
    """

    def __init__(self, target_population: int, offset: float = 0.25):
        """
        Initializes the SaRouletteWindowSelection operator.

        :param target_population: The number of individuals to select.
        :param offset: Adds constant value to each individual's score. Higher offset means
                       lower selection pressure. Offset should be non-negative.
        """

        self.target_population = target_population
        self.offset = offset

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
        max_val = max(args.population, key=lambda obj: obj.value).value
        min_val = min(args.population, key=lambda obj: obj.value).value
        maximize = args.evaluator.extremum() == GaExtremum.MAXIMUM
        div = max_val - min_val + self.offset
        sub = min_val - self.offset if maximize else max_val + self.offset

        while len(selected_individuals) < self.target_population:
            indiv = args.population[np.random.randint(0, pop_size)]
            diff = indiv.value - sub if maximize else sub - indiv.value
            weight = diff / div

            if np.random.rand() < weight:
                selected_individuals.append(indiv)

        return selected_individuals
