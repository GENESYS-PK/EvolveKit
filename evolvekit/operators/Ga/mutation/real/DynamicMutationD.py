import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs

class DynamicMutationD(GaOperator):
    def __init__(self, p_m: float = 0.1):
        """
        Initializes the dynamic (type D) smoothing mutation operator
        for real-valued chromosomes.

        :param p_m: Probability of applying smoothing to an individual.
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
        Applies the dynamic (type D) smoothing mutation to a population.

        :param args: ``GaOperatorArgs`` containing the current population.
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)

        if n < 2:
            return population

        new_population: List[GaIndividual] = []

        for individual in population:
            child = copy.deepcopy(individual)

            if np.random.rand() <= self.p_m:
                start = np.random.randint(0, n-1)
                stop = np.random.randint(start+1, n)

                source = child.real_chrom
                destination = source.copy()

                destination[start] = 0.67 * source[start] + 0.33 * source[start +  1]

                for i in range(start + 1, stop):
                    destination[i] = 0.25 * source[i - 1] + 0.5 * source[i] + 0.25 * source[i + 1]

                destination[stop] = 0.67 * source[stop] + 0.33 * source[stop - 1]

                child.real_chrom = destination

            new_population.append(child)

        return new_population