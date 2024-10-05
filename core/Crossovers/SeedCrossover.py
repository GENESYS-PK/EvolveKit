import numpy as np

from core import Crossover, Population
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class SeedCrossover(Crossover):
    """
    A class that implements Seed Crossover.

    In this crossover method, one individual acts as a "seed" parent, contributing more strongly to the child's genes.
    A specified ratio controls how much each parent contributes to the child's chromosome.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param seed_ratio: The ratio at which the seed parent's genes are weighted in the offspring, must be between 0 and 1.
    :raises ValueError: If the population size is less than 2 or if the seed_ratio is not between 0 and 1.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self, how_many_individuals: int, probability: float = 0, seed_ratio: float = 0.7
    ):
        super().__init__(how_many_individuals, probability)
        if not (0 <= seed_ratio <= 1):
            raise ValueError("The seed_ratio parameter must be between 0 and 1.")
        self.seed_ratio = seed_ratio

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always 2 children).
        """

        # Check probability of performing crossover
        if np.random.rand() > self.probability:
            return population_parent  # Return the same population if no crossover is performed

        # Select two different individuals from the population
        individual_index1 = np.random.randint(population_parent.population_size)
        individual_index2 = np.random.randint(population_parent.population_size)

        while individual_index1 == individual_index2:
            individual_index2 = np.random.randint(population_parent.population_size)

        def seed_crossover(g1: float, g2: float) -> float:
            # The seed parent's gene (g1) is weighted more strongly based on the seed_ratio
            return self.seed_ratio * g1 + (1 - self.seed_ratio) * g2

        # Lambda to return an iterator for gene pairs from two selected individuals
        tmp_zip = lambda: zip(
            population_parent.population[individual_index1].chromosome,
            population_parent.population[individual_index2].chromosome,
        )

        # Create two offspring, with parent 1 acting as the "seed" parent
        child1 = Individual(
            chromosome=np.array(
                [seed_crossover(g1, g2) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )
        child2 = Individual(
            chromosome=np.array(
                [seed_crossover(g2, g1) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )

        return Population(population=[child1, child2])
