import numpy as np

from core import Crossover, Population
from core.Crossover import Crossover
from core.Population import Population
from core.Representation import Representation


class MultipleCrossoverPerCouple(Crossover):
    """
    A class that implements Multiple Crossover Per Couple.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param max_number_of_crossovers: The maximum number of crossovers to perform on a single pair of individuals, must be int and greater than 0.
    :param crossover_function: Any crossover function that supports Representation.REAL, that is not a MultipleCrossoverPerCouple,
        which can work on Population of 2 Individuals and which returns a Population of 1 Individual.
    :raises ValueError: If some of the invalid crossover_function arguments are passed or the population size is less than 2.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        probability: float = 0,
        max_number_of_crossovers: int = 2,
        crossover_function: Crossover = None,
    ):
        super().__init__(how_many_individuals, probability)
        if int(max_number_of_crossovers) <= 0:
            raise ValueError(
                "The max_number_of_crossovers parameter must be greater than 0."
            )
        if crossover_function is None:
            raise ValueError("The crossover_function parameter must be passed.")
        elif not issubclass(type(crossover_function), Crossover):
            raise ValueError(
                "The crossover_function parameter must be of Crossover type."
            )
        elif isinstance(crossover_function, MultipleCrossoverPerCouple):
            raise ValueError(
                "The crossover_function parameter must not be a MultipleCrossoverPerCouple."
            )
        elif Representation.REAL not in crossover_function.allowed_representation:
            raise ValueError(
                "The crossover_function parameter must support Representation.REAL."
            )
        elif crossover_function.how_many_individuals != 1:
            raise ValueError(
                "The crossover_function parameter must return Population of 1 Individual."
            )

        self.max_number_of_crossovers = max_number_of_crossovers
        self.crossover_function = crossover_function

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always the same size as max_number_of_crossovers).
            If the returned population size is bigger than the how_many_individuals class field,
            then the combined offspring population will be truncated in the public method cross.
            In the original algorithm, the crossover ends while the combined offspring population is full,
            but since we don't know the exact number of combined offspring individuals in the public cross method,
            we can't check it here, and we have to return all of them and rely on the public cross method to truncate them,
            which will have the same outcome.
        """
        individual_index1 = np.random.randint(population_parent.population_size)
        individual_index2 = np.random.randint(population_parent.population_size)

        while individual_index1 == individual_index2:
            individual_index2 = np.random.randint(population_parent.population_size)

        # the crossover function should work on Population of 2 Individuals, and hopefully not edit the parent population at all
        tmp_population = Population(
            population=[
                population_parent.population[individual_index1],
                population_parent.population[individual_index2],
            ]
        )

        children = Population(population=[])
        for j in range(self.max_number_of_crossovers):
            children.add_to_population(self.crossover_function.cross(tmp_population))

        return children
