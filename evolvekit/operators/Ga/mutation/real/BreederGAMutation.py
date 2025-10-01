import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class BreederGAMutation(GaOperator):
    def __init__(self, p_m: float = 0.1, k: float = 5.0):
        """
        Initializes the Breeder GA (BGA) mutation operator for
        real-valued chromosomes.

        :param p_m: Probability of mutating a single gene.
        :type p_m: float
        :param k: Precision parameter of the BGA method — larger values
         produce smaller expected mutation steps.
        :type k: float
        """
        self.p_m = float(p_m)
        self.k = float(k)

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify
        its type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued
        mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Applies the Breeder GA (BGA) mutation operator to a given
        population of individuals.

        :param args: ``GaOperatorArgs`` containing the current population
         and an evaluator providing ``real_domain()``.
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)
        domain = args.evaluator.real_domain()

        new_population: List[GaIndividual] = []

        for individual in population:
            child = copy.deepcopy(individual)

            for i in range(n):
                if np.random.rand() < self.p_m:
                    low, up = domain[i]
                    gene_range = up - low
                    alpha = np.random.rand()
                    step = gene_range * (2.0 ** (-self.k * alpha))
                    if np.random.rand() <= 0.5:
                        new_value = child.real_chrom[i] - step
                    else:
                        new_value = child.real_chrom[i] + step

                    child.real_chrom[i] = new_value

            new_population.append(child)

        return new_population