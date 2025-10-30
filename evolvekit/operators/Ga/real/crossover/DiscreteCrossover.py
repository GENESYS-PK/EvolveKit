from typing import List
import sys

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DiscreteCrossover(GaOperator):
    def category(self):
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs discrete crossover on two randomly selected parents
        from the population.

        :param args: GaOperatorArgs containing population and evaluator
        :returns: One child created from two randomly selected parents
        """
        parent_1, parent_2 = np.random.choice(args.population, 2, replace=False)
        child = GaIndividual()
        alpha = np.random.uniform(0, sys.float_info.epsilon)
        if alpha <= 0.5:
            child.real_chrom = parent_1.real_chrom
        else:
            child.real_chrom = parent_2.real_chrom
        return [child]
