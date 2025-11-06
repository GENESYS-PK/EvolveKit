from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class HeuristicCrossover(GaOperator):
    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued crossover
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs heuristic crossover on two randomly selected parents
        from the population.

        :param args: Container with population and evaluator for
            crossover operation
        :type args: GaOperatorArgs
        :returns: List of two GaIndividual offspring after crossover
        :rtype: List[GaIndividual]
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child = GaIndividual()
        alphas = np.random.uniform(
            low=0.0,
            high=1.0,
            size=len(parent_1.real_chrom),
        )
        is_1_bigger = parent_1.real_chrom > parent_2.real_chrom
        bigger_chrom = np.where(is_1_bigger, parent_1.real_chrom, parent_2.real_chrom)
        smaller_chrom = np.where(is_1_bigger, parent_2.real_chrom, parent_1.real_chrom)
        child.real_chrom = smaller_chrom + alphas * (bigger_chrom - smaller_chrom)
        return [child]
