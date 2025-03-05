import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover

class TwoParentCrossover(Crossover):
    """
    Implements the arithmetical two-parent crossover algorithm.

    :param how_many_individuals: Number of individuals to create in the offspring.
    :param probability: Probability of performing the crossover.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0):
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform the arithmetical two-parent crossover operation.

        :param population_parent: The population from which parents are selected.
        :returns: The offspring population.
        """

        parent_1, parent_2 = np.random.choice(population_parent.population, 2)
        
        size = min(len(parent_1), len(parent_2))
        offspring1, offspring2 = [], []
        
        alpha = np.random.uniform(0, 1)  
        
        for i in range(size):
            new_xi = alpha * parent_1[i] + (1 - alpha) * parent_2[i]
            new_yi = alpha * parent_2[i] + (1 - alpha) * parent_1[i]
            
            offspring1.append(new_xi)
            offspring2.append(new_yi)
        
        return Population(
            population=[
                Individual(chromosome=offspring1, value=0),
                Individual(chromosome=offspring2, value=0),
            ]
        )
