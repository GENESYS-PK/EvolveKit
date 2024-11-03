import numpy as np
from abc import ABC, abstractmethod
from core.Population import Population
from core.Individual import Individual

class Mutation(ABC):
    """
    An abstract base class that serves as a blueprint for specific mutation methods.

    :param probability: The probability of performing the mutation operation.
    :param allowed_representation: A list of allowed representations for the mutation operation.
    """
    allowed_representation = []

    def __init__(self, probability: float = 0):
        self.probability = probability

    def mutate(self, population_parent: Population) -> Population:
        """
        Perform the mutation operations for the entire population.

        :param population_parent: The population to perform the mutation operation on.
        :returns: The mutated population.
        """
        for individual in population_parent.population:
            if np.random.rand() < self.probability:
                self._mutate(individual, population_parent)

        return population_parent

    @abstractmethod
    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Abstract method that defines the logic for performing mutation on an individual.

        :param individual: The individual to perform the mutation operation on.
        :param population: The population to which the individual belongs.
        :returns: None
        """
        pass
