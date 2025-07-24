import numpy as np
from typing import List

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga import GaState


class OnePointCrossover(GaOperator):
    def __init__(self):
        """
        Initializes the OnePointCrossover operator.

        This operator performs one-point crossover on binary-valued
        chromosomes.
        """
        pass

    def initialize(self, state: GaState):
        """
        Initializes the OnePointCrossover operator with the current state.

        This method is called before performing the crossover operation.

        Args:
            state (GaState): The current state of the genetic algorithm.
        """
        self.state = state

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
        chromosome_1 = n[0]
        chromosome_2 = n[1]
        binary_parent1 = np.unpackbits(chromosome_1.bin_chrom)
        binary_parent2 = np.unpackbits(chromosome_2.bin_chrom)
        crossover_point = np.random.randint(1, len(binary_parent1))

        binary_offspring1 = np.concatenate(
            (binary_parent1[:crossover_point], binary_parent2[crossover_point:])
        )
        binary_offspring2 = np.concatenate(
            (binary_parent2[:crossover_point], binary_parent1[crossover_point:])
        )

        chromosome_1.bin_chrom = np.packbits(binary_offspring1)
        chromosome_2.bin_chrom = np.packbits(binary_offspring2)
        offspring_population = [chromosome_1, chromosome_2]
        return offspring_population
