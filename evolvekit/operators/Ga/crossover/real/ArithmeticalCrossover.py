import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual


class ArithmeticalCrossover(GaOperator):
    def __init__(self, k: int = 2):
        """Initializes ArthmeticalCrossover operator

        :param k: number of parents to use in crossover.
        :type k: int
        """
        self.k = k

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator.

        :returns: GaOpCategory indicating real crossover operator type
        """
        return GaOpCategory.REAL_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """Performs arithmetical crossover on selected parents to create
        offspring.

        :param args: GaOperatorArgs containing population and other
            operator arguments
        :returns: List of GaIndividual offspring created through
            arithmetical crossover
        """
        offspring = []
        population = args.population
        random_values = np.random.uniform(size=self.k)
        alphas = random_values / np.sum(random_values)
        parents = np.random.choice(population, size=self.k, replace=False)
        for i in range(self.k):
            child = copy.deepcopy(parents[i])
            child.real_chrom = sum(
                alpha * parent.real_chrom for alpha, parent in zip(alphas, parents)
            )
            offspring.append(child)
        return offspring
