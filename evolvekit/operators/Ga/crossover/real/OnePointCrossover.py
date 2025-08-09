import numpy as np
from typing import List
import copy

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual


class OnePointCrossover(GaOperator):
    def __init__(self):
        """
        Initializes the OnePointCrossover operator.
        This operator performs one-point crossover on real-valued
        chromosomes.
        """
        pass

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs one-point crossover on two randomly selected parents.

        :param args: GaOperatorArgs containing the current population
        :returns: list of two GaIndividual offspring after crossover
        :raises ValueError: if population has fewer than two individuals
        """
        n = np.random.choice(args.population, 2, replace=False)
        parent_1 = n[0]
        parent_2 = n[1]
        offspring_1 = copy.deepcopy(parent_1)
        offspring_2 = copy.deepcopy(parent_2)
        crossover_point = np.random.randint(1, len(parent_1.real_chrom))
        offspring_1.real_chrom[:crossover_point] = \
            parent_1.real_chrom[:crossover_point]
        offspring_2.real_chrom[:crossover_point] = \
            parent_2.real_chrom[:crossover_point]
        offspring_1.real_chrom[crossover_point:] = \
            parent_2.real_chrom[crossover_point:]
        offspring_2.real_chrom[crossover_point:] = \
            parent_1.real_chrom[crossover_point:]
        return [offspring_1, offspring_2]
