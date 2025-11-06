from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class HeuristicCrossover2(GaOperator):
    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued crossover
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs heuristic crossover 2 on two randomly selected
        parents from the population.

        :param args: Container with population and evaluator for
            crossover operation
        :type args: GaOperatorArgs
        :returns: One GaIndividual offspring after crossover
        :rtype: List[GaIndividual]
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        alpha = np.random.uniform()
        child = GaIndividual()
        if parent_1.value >= parent_2.value:
            child.real_chrom = (
                alpha * (parent_2.real_chrom - parent_1.real_chrom)
                + parent_2.real_chrom
            )
        else:
            child.real_chrom = (
                alpha * (parent_1.real_chrom - parent_2.real_chrom)
                + parent_1.real_chrom
            )
        return [child]
