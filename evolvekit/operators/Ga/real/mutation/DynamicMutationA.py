import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DynamicMutationA(GaOperator):
    def __init__(self, p_m: float = 0.1):
        """
        Initializes the dynamic (type A) mutation operator for
        real-valued chromosomes.

        :param p_m: Probability of applying mutation to an individual.
        :type p_m: float
        """
        self.p_m = float(p_m)

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
        Applies the dynamic (type A) mutation to a given population.

        :param args: ``GaOperatorArgs`` containing the population
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

            if np.random.rand() < self.p_m:
                lam = np.random.randint(0, n)
                low, up = domain[lam]
                x = child.real_chrom[lam]

                alpha = np.random.normal(0.0, 1.0)
                s = 1.0 - np.exp(-abs(alpha))

                if np.random.rand() < 0.5:
                    x_new = x + (up - x) * s
                else:
                    x_new = x - (x - low) * s

                child.real_chrom[lam] = x_new

            new_population.append(child)

        return new_population
