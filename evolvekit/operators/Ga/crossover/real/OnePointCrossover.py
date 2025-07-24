import numpy as np
from typing import List
import copy

from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga import GaState


class OnePointCrossover(GaOperator):
    def __init__(self):
        """
        Initializes the OnePointCrossover operator.

        This operator performs one-point crossover on real-valued
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
            GaOpCategory: The operator category indicating real crossover.
        """
        return GaOpCategory.REAL_CROSSOVER

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
        n = np.random.choice(args.population, 2, replace=False)
        parent_1 = n[0]
        parent_2 = n[1]
        offspring_1 = copy.deepcopy(parent_1)
        offspring_2 = copy.deepcopy(parent_2)
        crossover_point = np.random.randint(1, len(parent_1.real_chrom))
        offspring_1.real_chrom[:crossover_point] = \
            parent_1.real_chrom[:crossover_point]
        offspring_2.real_chrom[:crossover_point] = \
            parent_2.real_chrom[:crossover_point]
        offspring_1.real_chrom[crossover_point:] = \
            parent_2.real_chrom[crossover_point:]
        offspring_2.real_chrom[crossover_point:] = \
            parent_1.real_chrom[crossover_point:]
        return [offspring_1, offspring_2]
