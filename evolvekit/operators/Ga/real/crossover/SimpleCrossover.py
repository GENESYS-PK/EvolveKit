from typing import List
import sys

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class SimpleCrossover(GaOperator):
    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs simple crossover on two randomly selected parents
        from the population.

        :param args: Container with population and evaluator for
            mutation operation
        :type args: GaOperatorArgs
        :returns: List of 2 two GaIndividual offspring after crossover
        :rtype: List[GaIndividual]
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child_1, child_2 = GaIndividual(), GaIndividual()
        size_of_chrom = len(parent_1.real_chrom)
        cross_point = np.random.randint(1, size_of_chrom + 1)
        child_1.real_chrom = parent_1.real_chrom[:cross_point]
        child_2.real_chrom = parent_2.real_chrom[:cross_point]
        if cross_point < size_of_chrom:
            alphas = np.random.uniform(
                low=0.0,
                high=1.0 + sys.float_info.epsilon,
                size=size_of_chrom - cross_point,
            )
            one_minus_alphas = 1 - alphas
            child_1.real_chrom = np.concatenate(
                (
                    child_1.real_chrom,
                    alphas * parent_2.real_chrom[cross_point:]
                    + one_minus_alphas * parent_1.real_chrom[cross_point:],
                )
            )
            child_2.real_chrom = np.concatenate(
                (
                    child_2.real_chrom,
                    alphas * parent_1.real_chrom[cross_point:]
                    + one_minus_alphas * parent_2.real_chrom[cross_point:],
                ),
            )
        return [child_1, child_2]
