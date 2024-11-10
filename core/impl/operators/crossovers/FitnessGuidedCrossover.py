import numpy as np
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation
from core import Individual, Population
from core.operators import Crossover


class FitnessGuidedCrossover(Crossover):
    """
    Implements fitness-guided crossover method based on the fitness values of the parents.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param beta: Defines the range for alpha (-beta ≤ alpha ≤ beta).
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self, how_many_individuals: int, probability: float = 0, beta: float = 0.5
    ):
        super().__init__(how_many_individuals, probability)
        self.beta = abs(beta)

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent (Population): The population to perform the crossover operation on.
        :returns: The offspring.
        """
        while True:
            parent_1, parent_2 = np.random.choice(population_parent.population, 2)
            if parent_1 != parent_2:
                break

        p1_chromosome = parent_1.chromosome
        p2_chromosome = parent_2.chromosome

        alpha = np.random.uniform(-self.beta, self.beta)

        if parent_1.value < parent_2.value:
            child_chromosome = [
                p1_chromosome[i] + alpha * (p1_chromosome[i] - p2_chromosome[i])
                for i in range(len(p1_chromosome))
            ]
        else:
            child_chromosome = [
                p2_chromosome[i] + alpha * (p2_chromosome[i] - p1_chromosome[i])
                for i in range(len(p1_chromosome))
            ]

        return Population(population=[Individual(chromosome=child_chromosome, value=0)])
