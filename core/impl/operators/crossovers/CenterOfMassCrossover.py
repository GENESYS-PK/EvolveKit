import numpy as np

from core import Population
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class CenterOfMassCrossover(Crossover):
    """
    A class that implements Center of Mass Crossover.

    In this crossover method, each gene of the child is created by averaging
    the genes of two parents, effectively finding the "center of mass" between them.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :raises ValueError: If the population size is less than 2.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0):
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always 2 children).
        """

        individual_index1 = np.random.randint(population_parent.population_size)
        individual_index2 = np.random.randint(population_parent.population_size)

        while individual_index1 == individual_index2:
            individual_index2 = np.random.randint(population_parent.population_size)

        def center_of_mass(g1: float, g2: float) -> float:
            # Compute the average (center of mass) between two genes
            return (g1 + g2) / 2.0

        # Lambda to return an iterator for gene pairs from two selected individuals
        tmp_zip = lambda: zip(
            population_parent.population[individual_index1].chromosome,
            population_parent.population[individual_index2].chromosome,
        )

        # Create two offspring by averaging the genes from two parents
        child1 = Individual(
            chromosome=np.array(
                [center_of_mass(g1, g2) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )
        child2 = Individual(
            chromosome=np.array(
                [center_of_mass(g1, g2) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )

        return Population(population=[child1, child2])
