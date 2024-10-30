import numpy as np
from core import Crossover, Individual, Population
from core.Population import Population
from core import FitnessFunction
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
        parent_1, parent_2 = np.random.sample(population_parent.population, 2)

        while True:
            alfa = np.random.uniform(0, 1)
            temporary_vector = (
                alfa * parent_1.chromosome + (1 - alfa) * parent_2.chromosome
            )
            if (
                self.fittness_function(*temporary_vector)
                < alfa * (parent_2.value - parent_1.value) + parent_1.value
            ):
                break

        a = (
            (1 - alfa) * parent_1.value
            + alfa * parent_2.value
            - self.fittness_function(*temporary_vector)
        ) / (alfa * (1 - alfa))
        b = (
            self.fittness_function(*temporary_vector)
            - parent_1.value
            + pow(alfa, 2) * (parent_1.value - parent_2.value)
        ) / (alfa * (1 - alfa))
        beta = -b / (2 * a)
        offspring = beta * parent_1.chromosome + (1 - beta) * parent_2.chromosome

        return Population(population=[offspring])
