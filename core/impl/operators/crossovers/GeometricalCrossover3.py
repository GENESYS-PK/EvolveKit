import numpy as np
import math
import random

from core import Population
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class GeometricalCrossover3(Crossover):
    """
    A class that implements Geometrical Crossover version 3 (Algorytmy genetyczne: kompendium. T. 1 strona 227)

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param k: The number of individuals to draft from population for crossover needs
    :raises ValueError: If population size is less than 2 or k parameter is wrong.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0, k: int = 2):
        super().__init__(how_many_individuals, probability)
        self.k = k

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always 1 child).
        """

        if self.k < 2 or self.k > population_parent.population_size:
            raise ValueError(
                "The k parameter must be more than 2 and less than population_size"
            )

        # drafting k individuals
        individuals_indices = np.array(
            [i for i in range(population_parent.population_size)]
        )
        drafted_individuals = np.random.choice(individuals_indices, self.k)

        # not sure if this approach is right
        chance_crossover = np.round(np.random.uniform(0, 1), 2)
        if self.probability < chance_crossover <= 1:
            return Population()

        # creating alfa array
        floats = np.array([random.random() for _ in range(len(drafted_individuals))])
        sum_floats = np.sum(floats)
        alfa = np.array([x / sum_floats for x in floats])

        chromosome_size = len(population_parent.population[
            drafted_individuals[0]
        ].chromosome)
        child_chromosome = np.zeros(chromosome_size)

        for i in range(chromosome_size):
            new_ind_val = 1
            for k in range(drafted_individuals.size):
                new_ind_val *= math.pow(
                    population_parent.population[drafted_individuals[k]].chromosome[i],
                    alfa[k],
                )
            child_chromosome[i] = new_ind_val

        child = Individual(chromosome=child_chromosome, value=0)

        return Population(population=[child])
