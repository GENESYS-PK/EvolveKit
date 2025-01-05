import numpy as np

from core import Population, Individual
from core.operators import Crossover
from core.Population import Population
from core.Representation import Representation


class SimulatedBinaryCrossover(Crossover):
    """
    Implements Simulated Binary Crossover. This crossover has similar search power to single-point crossover
    in the case of binary encoding.

     Algorytmy Genetyczne kompedium tom I Operator krzyżowania dla problemów numerycznych. Tomasz Gwiazda pp. 209.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param offspring_distance: Changes in the value of this parameter allow you to determine the distance of offsprings
                               from their parents. High values of the parameter increase the probability that the created
                               offspring will be close to the parents, while low values reduce this probability.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self,
        how_many_individuals: int,
        offspring_distance: float,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.offspring_distance = offspring_distance

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring.
        """
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace = False)

        if not isinstance(parent_1.chromosome, np.ndarray) or not isinstance(parent_2.chromosome, np.ndarray):

            parent_1_chromosome = np.array(parent_1.chromosome)
            parent_2_chromosome = np.array(parent_2.chromosome)
        else:
            parent_1_chromosome = parent_1.chromosome
            parent_2_chromosome = parent_2.chromosome

        alfa = np.random.uniform(0, 1)

        if alfa <= 0.5:
            base = 2 * alfa
        else:
            base = 1 / (2*(1 - alfa))

        exponent = 1 / (self.offspring_distance + 1)
        beta = pow(base,exponent)

        offspring_1_chromosome = 0.5 * ((1+beta) * parent_1_chromosome + (1-beta) * parent_2_chromosome)
        offspring_2_chromosome = 0.5 * ((1+beta) * parent_2_chromosome + (1-beta) * parent_1_chromosome)

        offspring_1 = Individual(offspring_1_chromosome)
        offspring_2 = Individual(offspring_2_chromosome)

        return Population([offspring_1, offspring_2])
