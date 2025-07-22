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
        This operator performs one-point crossover on real-valued chromosomes.
        """
        pass

    def initialize(self, state: GaState):
        """
        Initializes the OnePointCrossover operator with the current state.
        This method is called before performing the crossover operation.

        :param state: The current state of the genetic algorithm
        """
        self.state = state

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its type
        in the evolutionary algorithm framework.

        :returns: The operator category indicating real crossover
        """
        return GaOpCategory.REAL_CROSSOVER
    
    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs one-point crossover on two randomly selected 
        parents from the selected population.

        Args:
            args (GaOperatorArgs): Arguments containing the current state, 
            including the selected population.

        Returns:
            List[GaIndividual]: A list containing two offspring individuals
            resulting from the crossover operation.
        """
        offspring_population=[]
        population=args.state.selected_population
        n = np.random.choice(len(population), 2, replace=False)
        parent_1 = population[n[0]]
        parent_2 = population[n[1]]
        offspring_1 = copy.deepcopy(parent_1)
        offspring_2 = copy.deepcopy(parent_2)
        crossover_point = np.random.randint(0, len(parent_1.real_chrom) - 1)
        offspring_1.real_chrom[:crossover_point + 1] = parent_1.real_chrom[:crossover_point + 1]
        offspring_2.real_chrom[:crossover_point + 1] = parent_2.real_chrom[:crossover_point + 1]
        offspring_1.real_chrom[crossover_point + 1:] = parent_2.real_chrom[crossover_point + 1:]
        offspring_2.real_chrom[crossover_point + 1:] = parent_1.real_chrom[crossover_point + 1:]
        offspring_population.append(offspring_1)
        offspring_population.append(offspring_2)
        return offspring_population