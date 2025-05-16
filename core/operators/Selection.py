from abc import ABC, abstractmethod
from core.Population import Population


class Selection(ABC):
    """
    An abstract base class that serves as a blueprint for specific selection methods.

    :param target_population: The target population size.
    :param maximize: A boolean flag that indicates whether the selection method should maximize or minimize the fitness function.
    :param allowed_representation: A list of allowed representations for the selection operation.
    """

    allowed_representation = []

    def __init__(self, target_population: int, maximize: bool):
        self.target_population = target_population
        self.maximize = maximize

    def select(self, population: Population) -> Population:
        """
        Perform the selection operation on the population.

        :param population: The population to perform the selection operation on.
        :returns: The selected population.
        """
        selected_population = self._select(population)

        population.trim_children(self.target_population)
        return selected_population

    @abstractmethod
    def _select(self, population: Population) -> Population:
        """
        Abstract method that defines the logic for performing selection on a population.

        :param population: The population to perform the selection operation on.
        :returns: The selected population.
        """
        pass
