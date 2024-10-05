import numpy as np
import random
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation
from core import Crossover


class SetOrientedCrossover(Crossover):
    """
    Implements set-oriented crossover method where genes of offspring are determined by parent gene equality
    and random values from a specified range for each gene.

    Algorytmy genetyczne : kompendium. T. 1, Operator krzyżowania dla problemów numerycznych. / T. D. Gwiazda, 2021 (strona 275)

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param ranges: List of tuples representing the (min, max) range for each gene.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, ranges: list, probability: float = 0):
        super().__init__(how_many_individuals, probability)
        self.ranges = np.array(ranges)

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform crossover operation to generate offspring.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        while True:
            parent_1, parent_2 = random.sample(population_parent.population, 2)
            if parent_1 != parent_2:
                break

        p1_chromosome = np.array(parent_1.chromosome)
        p2_chromosome = np.array(parent_2.chromosome)
        child1_chromosome = []
        child2_chromosome = []

        for i in range(len(p1_chromosome)):
            if p1_chromosome[i] == p2_chromosome[i]:
                child1_chromosome.append(p1_chromosome[i])
                child2_chromosome.append(p1_chromosome[i])
            else:
                lower_bound, upper_bound = self.ranges[i]
                child1_chromosome.append(
                    round(np.random.uniform(lower_bound, upper_bound), 2)
                )
                child2_chromosome.append(
                    round(np.random.uniform(lower_bound, upper_bound), 2)
                )

        return Population(
            population=[
                Individual(chromosome=child1_chromosome, value=0),
                Individual(chromosome=child2_chromosome, value=0),
            ]
        )
