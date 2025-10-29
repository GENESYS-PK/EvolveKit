from typing import List

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class AverageCrossover(GaOperator):
    def category(self):
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs averrage crossover on two randomly selected parents
        from the population.

        :param args: GaOperatorArgs containing population and evaluator
        :returns: One child createdfrom two randomly selected parents
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child = GaIndividual()
        child.real_chrom = [
            (chrom1 + chrom2) / 2
            for chrom1, chrom2 in zip(parent_1.real_chrom, parent_2.real_chrom)
        ]
        return [child]
