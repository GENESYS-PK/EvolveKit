import numpy as np
from contourpy.types import offset_dtype

from core import Population, Individual
from core.operators import Crossover
from core.Population import Population
from core.fitness_function import FitnessFunction
from core.Representation import Representation


class FitnessBasedParabolicCrossover(Crossover):
    """
    Implements Fitness-Based Parabolic Crossover method in which offspring is a vertex of parabola.

     Algorytmy Genetyczne kompedium tom I Operator krzyżowania dla problemów numerycznych. Tomasz Gwiazda pp. 367.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param fittness_function: A function that is being optimized.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fittness_function: FitnessFunction,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fittness_function = fittness_function

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

        if not np.allclose(parent_1_chromosome, parent_2_chromosome):

            while True:
                alfa = np.random.uniform(0, 1)
                temporary_vector = (
                    alfa * parent_1_chromosome + (1 - alfa) * parent_2_chromosome
                )
                if (
                    self.fittness_function.fitness_function(temporary_vector)
                    < alfa * (parent_2.value - parent_1.value) + parent_1.value
                ):
                    break

            a = (
                (1 - alfa) * parent_1.value
                + alfa * parent_2.value
                - self.fittness_function.fitness_function(temporary_vector)
            ) / (alfa * (1 - alfa))
            b = (
                self.fittness_function.fitness_function(temporary_vector)
                - parent_1.value
                + pow(alfa, 2) * (parent_1.value - parent_2.value)
            ) / (alfa * (1 - alfa))
            beta = -b / (2 * a)
            offspring_chromosome = beta * parent_1_chromosome + (1 - beta) * parent_2_chromosome
            offspring = Individual(offspring_chromosome)

            return Population(population=[offspring])

        else:

            return Population(population=[parent_1])
