from typing import List

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class FlatCrossover(GaOperator):
    def category(self):
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs flat crossover  on two randomly selected parents
        from the population.

        :param args: GaOperatorArgs containing population and evaluator
        :returns: two children created from two randomly selected
            parents
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child = GaIndividual()
        low_boundary = np.minimum(
            parent_1.real_chrom, np.nextafter(parent_2.real_chrom, np.inf)
        )
        high_boundary = np.maximum(
            parent_1.real_chrom, np.nextafter(parent_2.real_chrom, np.inf)
        )
        child.real_chrom = np.random.uniform(low=low_boundary, high=high_boundary)
        return [child]
