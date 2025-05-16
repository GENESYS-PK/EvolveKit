import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function.FitnessFunction import FitnessFunction


class PositionCrossover(Crossover):
    """
    Implements position-based crossover method.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fitness_function: A callable fitness function that evaluates individuals.
    :param a: The adjustment value for the chromosomes during crossover.
    :param B: The adjustment value for the chromosomes during crossover.
    :param maximize: Whether to maximize or minimize the fitness function.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        a: float = 1.0,
        B: float = 1.0,
        maximize: bool = True,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function
        self.a = a
        self.B = B
        self.maximize = maximize

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform position-based crossover for the entire population and return the offspring population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """

        parent_1, parent_2 = np.random.choice(
            population_parent.population, size=2, replace=False
        )

        size = min(parent_1.chromosome.size, parent_2.chromosome.size)
        ind3, ind4 = [], []

        find1 = self.fitness_function.fitness_function(parent_1.chromosome)
        find2 = self.fitness_function.fitness_function(parent_2.chromosome)

        if (self.maximize and find1 >= find2) or (not self.maximize and find1 <= find2):
            for i in range(size):
                ind3.append(parent_1.chromosome[i])
                ind4.append(parent_1.chromosome[i])
        else:
            for i in range(size):
                ind3.append(parent_2.chromosome[i])
                ind4.append(parent_2.chromosome[i])

        for i in range(size):
            if np.random.random() < 0.5:
                ind3[i] -= self.a
                ind4[i] += self.B
            else:
                ind3[i] += self.a
                ind4[i] -= self.B

        return Population(
            population=[
                Individual(chromosome=ind3, value=0),
                Individual(chromosome=ind4, value=0),
            ]
        )
