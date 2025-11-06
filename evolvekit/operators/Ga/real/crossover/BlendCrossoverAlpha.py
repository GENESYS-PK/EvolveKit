from typing import List

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class BlendCrossoverAlpha(GaOperator):
    def __init__(self, alpha: float = 0.3):
        """
        Initializes BlendCrossoverAlpha operator.

        :param alpha: parameter alpha.
        :type alpha: int
        """
        self.alpha = alpha

    def category(self):
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs blend crossover on two randomly selected parents
        from the population.

        :param args: GaOperatorArgs containing population and evaluator
        :returns: two children created from two randomly selected
            parents
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child_1 = GaIndividual()
        child_2 = GaIndividual()
        parent_1_chromosomes = np.array(parent_1.real_chrom)
        parent_2_chromosomes = np.array(parent_2.real_chrom)
        delta = np.abs(parent_1_chromosomes - parent_2_chromosomes)
        low_boundary = (
            np.minimum(parent_1_chromosomes, parent_2_chromosomes) - self.alpha * delta
        )
        high_boundary = (
            np.maximum(parent_1_chromosomes, parent_2_chromosomes) + self.alpha * delta
        )
        u = np.random.uniform(low=low_boundary, high=high_boundary)
        child_1.real_chrom = u
        u = np.random.uniform(low=low_boundary, high=high_boundary)
        child_2.real_chrom = u
        return [child_1, child_2]
