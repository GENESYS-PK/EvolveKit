import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DynamicMutationC(GaOperator):
    def __init__(self, p_m: float = 0.1):
        """
        Initializes the dynamic (type C) shift mutation for
        real-valued chromosomes.

        :param p_m: Probability of applying the shift to an individual.
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
        Applies the dynamic (type C) shift mutation to a population.

        :param args: ``GaOperatorArgs`` containing the current population.
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)
        new_population: List[GaIndividual] = []

        for individual in population:
            child = copy.deepcopy(individual)

            if np.random.rand() <= self.p_m:
                pivot = np.random.randint(0, n)
                shift_right = np.random.rand() < 0.5

                source = child.real_chrom
                destination = source.copy()

                if shift_right:
                    for i in range(pivot + 1, n):
                        destination[i] = source[i - 1]
                else:
                    for i in range(0, pivot):
                        destination[i] = source[i + 1]

                child.real_chrom = destination

            new_population.append(child)

        return new_population
