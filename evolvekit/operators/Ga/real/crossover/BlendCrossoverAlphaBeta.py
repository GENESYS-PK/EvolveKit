from typing import List

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class BlendCrossoverAlphaBeta(GaOperator):
    def __init__(self, alpha: float = 0.3, beta: float = 0.5):
        """
        Initializes Blend Crossover Alpha Beta operator.

        :param alpha: Parameter alpha.
        :type alpha: int
        :param beta: Parameter beta.
        :type beta: int
        """
        self.alpha = alpha
        self.beta = beta

    def category(self):
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs blend crossover alpha beta on two randomly selected
        parents from the population.

        :param args: GaOperatorArgs containing population and evaluator
        :returns: two children created from two randomly selected
            parents
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child_1 = GaIndividual()
        child_2 = GaIndividual()
        parent_1_chromosomes = parent_1.real_chrom
        parent_2_chromosomes = parent_2.real_chrom
        is_1_smaller = parent_1_chromosomes <= parent_2_chromosomes
        delta = np.abs(parent_1_chromosomes - parent_2_chromosomes)
        low_boundary = (
            np.where(is_1_smaller, parent_1_chromosomes, parent_2_chromosomes)
            - np.where(is_1_smaller, self.alpha, self.beta) * delta
        )
        high_boundary = (
            np.where(is_1_smaller, parent_2_chromosomes, parent_1_chromosomes)
            + np.where(is_1_smaller, self.beta, self.alpha) * delta
        )
        u = np.random.uniform(low=low_boundary, high=high_boundary)
        child_1.real_chrom = u
        u = np.random.uniform(low=low_boundary, high=high_boundary)
        child_2.real_chrom = u
        return [child_1, child_2]
