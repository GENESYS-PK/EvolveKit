import numpy as np
from typing import List

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual


class OnePointCrossover(GaOperator):
    def __init__(self):
        """
        Initializes the OnePointCrossover operator.

        This operator performs one-point crossover on binary-valued
        chromosomes.
        """
        pass

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator.

        This is used to classify its type in the evolutionary algorithm
        framework.

        Returns:
            GaOpCategory: The operator category indicating binary crossover.
        """
        return GaOpCategory.BIN_CROSSOVER

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs one-point crossover on two randomly selected parents.

        Args:
            args (GaOperatorArgs): Arguments containing the current state,
                including the selected population.

        Returns:
            List[GaIndividual]: A list containing two offspring individuals
                resulting from the crossover operation.
        """
        population = args.population
        n = np.random.choice(population, 2, replace=False)
        individual_1 = n[0]
        individual_2 = n[1]
        binary_parent1 = np.unpackbits(individual_1.bin_chrom)
        binary_parent2 = np.unpackbits(individual_2.bin_chrom)
        crossover_point = np.random.randint(1, len(binary_parent1))

        binary_offspring1 = np.concatenate(
            (binary_parent1[:crossover_point], binary_parent2[crossover_point:])
        )
        binary_offspring2 = np.concatenate(
            (binary_parent2[:crossover_point], binary_parent1[crossover_point:])
        )

        individual_1.bin_chrom = np.packbits(binary_offspring1)
        individual_2.bin_chrom = np.packbits(binary_offspring2)
        offspring_population = [individual_1, individual_2]
        return offspring_population
