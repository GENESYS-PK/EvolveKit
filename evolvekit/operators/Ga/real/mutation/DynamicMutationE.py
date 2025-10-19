import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DynamicMutationE(GaOperator):
    def __init__(self, p_m: float = 0.1):
        """
        Initializes the dynamic (type E) swap mutation for
        real-valued chromosomes.

        :param p_m: Probability of applying the swap to an individual.
        :type p_m: float
        """
        self.p_m = p_m

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify
        its type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Applies the dynamic (type E) swap mutation to a population.

        :param args: ``GaOperatorArgs`` containing the current population.
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)
        new_population: List[GaIndividual] = []

        if n < 2:
            return population

        for individual in population:
            child = copy.deepcopy(individual)
            if np.random.rand() < self.p_m:
                lam = np.random.randint(0, n - 1)
                chromosome = child.real_chrom
                chromosome[lam], chromosome[lam + 1] = (
                    chromosome[lam + 1],
                    chromosome[lam],
                )
                child.real_chrom = chromosome

            new_population.append(child)

        return new_population
