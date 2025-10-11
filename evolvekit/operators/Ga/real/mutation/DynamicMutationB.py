import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DynamicMutationB(GaOperator):
    def __init__(self, p_m: float = 0.1, beta: float = 0.02):
        """
        Initializes the dynamic (type B) "creep" mutation operator
        for real-valued chromosomes.

        :param p_m: Probability of mutating a single gene.
        :type p_m: float
        :param beta: Scaling factor controlling mutation intensity.
        :type beta: float
        """
        self.p_m = p_m
        self.beta = beta

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
        Applies the dynamic (type B) creep mutation to a population.

        :param args: ``GaOperatorArgs`` with population and evaluator.
        :type args: GaOperatorArgs
        :returns: New population with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)
        domain = args.evaluator.real_domain()
        new_population: List[GaIndividual] = []

        for individual in population:
            child = copy.deepcopy(individual)

            for i in range(n):
                if np.random.rand() <= self.p_m:
                    low, up = domain[i]
                    width = up - low
                    if width <= 0:
                        continue

                    x = child.real_chrom[i]
                    alpha = np.random.normal(0.0, 1.0)

                    delta = alpha * self.beta * width
                    x_new = x + delta
                    gamma = abs(delta)

                    while x_new > up or x_new < low:
                        gamma *= 0.5
                        if x_new > up:
                            x_new = x + gamma
                        else:
                            x_new = x - gamma

                    child.real_chrom[i] = x_new

            new_population.append(child)

        return new_population
