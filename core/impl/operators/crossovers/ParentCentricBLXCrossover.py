import numpy as np
import math
import random

from core import Population
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class ParentCentricBLXCrossover(Crossover):
    """
    A class that implements ParentCentricBLX(alfa)Crossover (Algorytmy genetyczne: kompendium. T. 1 strona 334)
    Takes 2 parents and creates 1 child.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param interval: The interval (for every variable) for the crossover operation. [(lower, upper), (lower, upper)].
    :param probability: The probability of performing the crossover operation.
    :param alfa: The alpha parameter for the crossover operation.
    :raises ValueError: If population size is less than 2 or k parameter is wrong.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int,  interval: list[tuple], probability: float = 0, alfa=0.5):
        super().__init__(how_many_individuals, probability)
        self.interval = interval
        self.alfa = alfa

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always 1 child).
        """

        individual_index1 = np.random.randint(population_parent.population_size)
        individual_index2 = np.random.randint(population_parent.population_size)

        while individual_index1 == individual_index2:
            individual_index2 = np.random.randint(population_parent.population_size)

        # not sure if this approach is right
        chance_crossover = np.round(np.random.uniform(0, 1), 2)
        if self.probability < chance_crossover <= 1:
            return Population()

        chromosome_size = len(
            population_parent.population[individual_index1].chromosome
        )
        child_chromosome = np.zeros(chromosome_size)

        for i in range(chromosome_size):
            child_chromosome[i] = math.sqrt(
                (math.pow(population_parent.population[individual_index1].chromosome[i], 2)
                 + math.pow((population_parent.population[individual_index2].chromosome[i]), 2))
                / 2)

        child = Individual(chromosome=child_chromosome, value=0)

        return Population(population=[child])

        rnd = np.random.uniform(0, 1)
        if rnd <= 0.5:
            d1 = []
            for i in range(chromosome_size):
                d1.append(abs(population_parent.population[individual_index1].chromosome[i] - population_parent.population[individual_index2].chromosome[i][i]))
                u = np.random.uniform(max(self.interval[i][0], population_parent.population[individual_index1].chromosome[i][i] - alfa * d1[i]),
                                      min(self.interval[i][1], population_parent.population[individual_index1].chromosome[i][i] + alfa * d1[i]))  # interval[i][0] - lower, interval[i][1] - upper
                child_chromosome[i] = u
        else:
            d2 = []
            for i in range(chromosome_size):
                d2.append(abs(population_parent.population[individual_index1].chromosome[i][i] - population_parent.population[individual_index2].chromosome[i][i]))
                u = np.random.uniform(max(self.interval[i][0], population_parent.population[individual_index2].chromosome[i][i] - alfa * d2[i]),
                                      min(self.interval[i][1], population_parent.population[individual_index2].chromosome[i][i] + alfa * d2[i]))  # interval[i][0] - lower, interval[i][1] - upper
                child_chromosome[i] = u

        child = Individual(chromosome=child_chromosome, value=0)

        return Population(population=[child])
