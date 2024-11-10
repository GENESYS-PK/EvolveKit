import numpy as np
from abc import ABC, abstractmethod
from core.Population import Population


class Crossover(ABC):
    """
    An abstract base class that serves as a blueprint for specific crossover methods.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param allowed_representation: A list of allowed representations for the crossover operation.
    """

    allowed_representation = []

    def __init__(self, how_many_individuals: int, probability: float = 0):
        self.how_many_individuals = how_many_individuals
        self.probability = probability

    def cross(self, population_parent: Population) -> Population:
        """
        Perform the crossover operations for the entire population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        offspring = Population()

        while offspring.population_size < population_parent.population_size:
            if np.random.rand() < self.probability:
                offspring.add_to_population(self._cross(population_parent))

        offspring.trim_children(self.how_many_individuals)
        return offspring

    @abstractmethod
    def _cross(self, population_parent: Population) -> Population:
        """
         Abstract method that defines the logic for performing crossover on a parent population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        pass
