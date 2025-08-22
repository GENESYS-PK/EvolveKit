import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class BoundaryMutation(GaOperator):
    def __init__(self,  p_bm: float = 0.1):
        """
        Initializes the boundary mutation operator for
        real-valued chromosomes.

        With probability ``p_bm`` per individual, the operator picks
        one random gene and replaces its value with either the lower
        or the upper boundary of that gene's domain, where domains
        are obtained from the evaluator's ``real_domain()`` method.

        :param p_bm: Probability of applying boundary mutation to a
         given individual.
        :type p_bm: float
        """
        self.p_bm = float(p_bm)

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
        Applies the boundary mutation operator to a given
        population of individuals.

        For each individual, with probability ``p_bm``, a mutation
        index ``k`` is drawn uniformly from the set of gene positions.
        The gene value ``x_k`` is then overwritten with either the
        lower bound ``x_k^l`` or the upper bound ``x_k^u`` of its
        domain (chosen with equal probability). Bounds are read from
        ``args.evaluator.real_domain()``, which must return a list of
        tuples ``(low, up)`` for each gene.

        :param args: ``GaOperatorArgs`` containing the current population
         and an evaluator providing ``real_domain()``.
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals.
        :rtype: List[GaIndividual]
        """
        population = args.population
        n = len(population[0].real_chrom)
        domain = args.evaluator.real_domain()

        new_population = []

        for individual in population:
            child = copy.deepcopy(individual)

            if np.random.rand() <= self.p_bm:
                k = np.random.randint(0,n)
                low, up = domain[k]

                child.real_chrom[k] = low if np.random.rand() < 0.5 else up

            new_population.append(child)

        return new_population


