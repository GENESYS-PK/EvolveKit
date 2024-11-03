import numpy as np
import math
from core import Individual, Population, Representation
from core.operators import Crossover


class GaussCrossover(Crossover):
    """
    Implements Gauss crossover method based on the Gaussian mechanism.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fitness_function: A callable fitness function that evaluates individuals.
    :param xp: The lower bound for offspring values.
    :param xk: The upper bound for offspring values.
    :param probability: The probability of performing the crossover operation.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: callable,
        xp: float,
        xk: float,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function
        self.xp = xp
        self.xk = xk

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform Gaussian crossover for the entire population and return the offspring population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """

        parent_1, parent_2 = np.random.choice(population_parent.population, 2)

        size = min(len(parent_1), len(parent_2))
        offspring1, offspring2 = [], []

        for i in range(size):
            distance = math.fabs(parent_1[i] - parent_2[i])
            alpha = np.random.rand()

            if np.random.uniform(0, 1) <= 0.5:
                new_val1 = parent_1[i] + alpha * (distance / 3)
                new_val2 = parent_2[i] + alpha * (distance / 3)
            else:
                new_val1 = parent_2[i] + alpha * (distance / 3)
                new_val2 = parent_1[i] + alpha * (distance / 3)

            if new_val1 < self.xp or new_val1 > self.xk:
                new_val1 = parent_1[i]
            if new_val2 < self.xp or new_val2 > self.xk:
                new_val2 = parent_2[i]

        offspring1.append(new_val1)
        offspring2.append(new_val2)

        return Population(
            population=[
                Individual(chromosome=offspring1, value=0),
                Individual(chromosome=offspring2, value=0),
            ]
        )
