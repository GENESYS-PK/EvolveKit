import numpy as np
from core import Individual, Population
from core.operators import Mutation
from core.Representation import Representation

class DynamicMutationC(Mutation):
    """
    Version C – shift mutation. (Algorytmy genetyczne. Kompendium, t. 2, str. 198 - 199).

    This mutation selects a random shift point in the chromosome and shifts the values
    either to the left or to the right. The values before or after the shift point are
    preserved accordingly.

    :param probability: Probability of applying mutation to an individual.
    :returns: None – the individual's chromosome is mutated in place.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float = 1.0):
        super().__init__(probability)

    def _mutate(self, individual: Individual, population: Population) -> None:
        chromosome = individual.chromosome.copy()
        n = len(chromosome)

        q = np.random.randint(0, n)

        shift_right = np.random.uniform(0, 1) < 0.5

        new_chromosome = np.zeros_like(chromosome)

        if shift_right:
            for i in range(0, q + 1):
                new_chromosome[i] = chromosome[i]
            for i in range(q + 1, n):
                new_chromosome[i] = chromosome[i - 1]
        else:
            for i in range(0, q):
                new_chromosome[i] = chromosome[i + 1]
            for i in range(q, n):
                new_chromosome[i] = chromosome[i]

        individual.chromosome = new_chromosome
