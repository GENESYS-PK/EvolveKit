import numpy as np
from typing import List

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class UniformMutaion(GaOperator):
    def __init__(self, p_um):
        """Initializes uniform mutation operator for real-valued
        chromosomes.

        :param p_um: Propability of mutation for uniform mutation
        :type p_copy: float
        """
        self.p_um = p_um

    def category(self) -> GaOpCategory:
        """Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """Performs uniform mutation on real-valued chromosomes in the
        population.

        :param args: Container with population and evaluator for
            mutation operation
        :type args: GaOperatorArgs
        :returns: Population with potentially mutated individuals
        :rtype: List[GaIndividual]
        :raises AttributeError: If individual lacks real_chrom attribute
            or evaluator lacks real_domain method
        """
        population = args.population
        for individual in population:
            if (
                np.random.uniform(low=np.nextafter(0, 1), high=np.nextafter(1, 0))
                <= self.p_um
            ):
                muttation_point = np.random.randint(0, len(individual.real_chrom))
                lower_limit, upper_limit = args.evaluator.real_domain()[muttation_point]
                individual.real_chrom[muttation_point] = np.random.uniform(
                    low=lower_limit, high=upper_limit
                )
        return population
