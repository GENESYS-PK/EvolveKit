import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function import FitnessFunction


class BestBestCrossover(Crossover):
    """
    Implements Best-Best crossover method.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        fitness_function: FitnessFunction,
        how_many_individuals: int,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform Best-Best crossover for the entire population and return the offspring population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        # I have problem here how to get best ones... xelp if possible pls
        self.fitness_function.eval_population(population_parent)
        sorted_population = sorted(
            population_parent.population, key=lambda ind: ind.value, reverse=True
        )
        parent_1, parent_2 = sorted_population[:2]

        l = np.random.uniform(0, 1)
        m = np.random.uniform(0, 1)
        q = np.random.uniform(0, 1)
        j = np.random.uniform(0, 1)

        ind1, ind2 = parent_1.chromosome, parent_2.chromosome

        ind3 = []
        ind4 = []

        for i in range(min(int(l * len(ind1)), len(ind1))):
            ind3.append(ind1[i])
        for i in range(
            max(int(q * len(ind2)), 0), min(int(j * len(ind2)) + 1, len(ind2))
        ):
            ind3.append(ind2[i])
        for i in range(min(int(m * len(ind2)) + 1, len(ind2)), len(ind2)):
            ind3.append(ind2[i])

        for i in range(min(int(q * len(ind2)), len(ind2))):
            ind4.append(ind2[i])
        for i in range(
            min(int(l * len(ind1)), len(ind1)), min(int(m * len(ind1)) + 1, len(ind1))
        ):
            ind4.append(ind1[i])
        for i in range(min(int(j * len(ind2)) + 1, len(ind2)), len(ind2)):
            ind4.append(ind2[i])

        return Population(
            population=[
                Individual(chromosome=ind3, value=0),
                Individual(chromosome=ind4, value=0),
            ]
        )
