from typing import List
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class SaRouletteSigmaSelection(GaOperator):
    """
    Stochastic acceptance roulette-wheel selection with sigma scaling.

    For more details about stochastic acceptance roulette-wheel see:
    https://arxiv.org/abs/1109.3627

    For more details about sigma scaling see:
    https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Practical%20Handbook%20of%20GENETIC%20ALGORITHMS%2C%20Volume%20II/ganf3.pdf
    """

    def __init__(
        self,
        target_population: int,
        offset: float = 1.0,
        minimum: float = 0.1,
        factor: float = 2.0,
        epsilon: float = 0.0001,
    ):
        """
        Initializes the SaRouletteSigmaSelection operator.

        :param target_population: The number of individuals to select.
        :param offset: Adds constant value to each individual's score. Offset should be non-negative.
        :param minimum: Minimum score after scaling. Should be non-negative.
        :param factor: Constant factor in the denominator. Should be greater than zero.
        :param epsilon: Standard deviation threshold below which all individuals are treated as if
                        they have equal scores. This should be small value, but greater than zero.
        """

        self.target_population = target_population
        self.offset = offset
        self.minimum = minimum
        self.factor = factor
        self.epsilon = epsilon

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
        scores = np.array([obj.value for obj in args.population])
        maximize = args.evaluator.extremum() == GaExtremum.MAXIMUM
        stdev = args.statistics.stdev
        mean = args.statistics.mean

        if stdev < self.epsilon:
            while len(selected_individuals) < self.target_population:
                indiv = args.population[np.random.randint(0, pop_size)]
                selected_individuals.append(indiv)
            return selected_individuals

        c = 1.0 / (stdev * self.factor)
        maxDiff = max(scores) - mean if maximize else mean - min(scores)
        scale = maxDiff * c + self.offset

        while len(selected_individuals) < self.target_population:
            indiv = args.population[np.random.randint(0, pop_size)]
            diff = indiv.value - mean if maximize else mean - indiv.value
            weight = max(diff * c + self.offset, self.minimum)

            if np.random.rand() * scale < weight:
                selected_individuals.append(indiv)

        return selected_individuals
