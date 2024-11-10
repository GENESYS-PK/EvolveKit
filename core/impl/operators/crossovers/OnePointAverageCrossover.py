import numpy as np
from core import Population
from core.operators import Crossover
from core.Population import Population
from core.Representation import Representation


class OnePointAverageCrossover(Crossover):

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0):
        """
        Constructor for the OnePointAverageCrossover (1-PAX) class.

        :param how_many_individuals: The number of individuals to create in the offspring.
        :param probability: The probability of performing the crossover operation.
        """
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform One Point Average Crossover on the parent population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        :raises ValueError: If the population size is less than 2.
        """
        offspring_population = Population()

        # Randomly select two parents from the parent population
        parent_indices = np.random.choice(
            population_parent.population_size, 2, replace=False
        )
        parent_x = population_parent.population[parent_indices[0]].chromosome
        parent_y = population_parent.population[parent_indices[1]].chromosome

        # Crossover point - select a point for crossover
        shortest_parent_size = min(parent_x.size, parent_y.size)
        crossover_point = np.random.randint(0, shortest_parent_size)

        # Averaging at the crossover point
        new_gene = (parent_x[crossover_point] + parent_y[crossover_point]) / 2
        parent_x[crossover_point] = new_gene
        parent_y[crossover_point] = new_gene

        # Add new individuals to the offspring population
        offspring_population.add_to_population(parent_x)
        offspring_population.add_to_population(parent_y)
        return offspring_population
