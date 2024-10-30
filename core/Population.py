from dataclasses import dataclass, field
from typing import List, Iterable, Self
from core.Individual import Individual
import singledispatch


@dataclass
class Population:
    """
    Data class that represents a population of individuals.

    :param population: A list of individuals that make up the population. Defaults to an empty list.
    """

    population: List[Individual] = field(default_factory=list)

    @property
    def population_size(self) -> int:
        """
        Get the size of the population.

        :returns: The number of individuals in the population.
        """
        return len(self.population)

    @singledispatch
    def add_to_population(self, individual: Individual) -> None:
        """
        Add an individual to the population.

        :param individual: The individual to add to the population.
        :returns: None
        """
        self.population.append(individual)

    @singledispatch
    def add_to_population(self, individual_list: Iterable[Individual]) -> None:
        """
        Add a list of individuals to the population.

        :param individual_list: A list of individuals to add to the population.
        :returns: None
        """
        self.population.extend(individual_list)

    @singledispatch
    def add_to_population(self, individual_list: Self) -> None:
        """
        Add members of another population to the current population.

        :param individual_list: Another population to add to the current population.
        :returns: None
        """
        self.population.extend(individual_list.population)

    def trim_children(self, how_many_keep: int) -> None:
        """
        Trim excess individuals from the population.

        :param how_many_keep: The number of individuals to keep in the population.
        :returns: None
        """
        self.population = self.population[:how_many_keep]
