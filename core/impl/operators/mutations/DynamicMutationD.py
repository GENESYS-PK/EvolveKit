import numpy as np
from core import Individual, Population
from core.operators import Mutation
from core.Representation import Representation

class DynamicMutationD(Mutation):
    """
    Version D – smoothing mutation. (Algorytmy genetyczne. Kompendium, t. 2, str. 199).

    This mutation selects a range within the chromosome and smooths the gene values
    using a weighted average of neighboring values. 

    :param probability: Probability of applying the mutation to an individual.
    :returns: None – the individual's chromosome is modified in place.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float = 1.0):
        super().__init__(probability)

    def _mutate(self, individual: Individual, population: Population) -> None:
        if np.random.uniform(0, 1) > self.probability:
            return

        chromosome = individual.chromosome.copy()
        n = len(chromosome)

        start = np.random.randint(0, n - 1)
        stop = np.random.randint(start + 1, n)

        new_chromosome = chromosome.copy()

        if start + 1 < n:
            new_chromosome[start] = (
                0.67 * chromosome[start] + 0.33 * chromosome[start + 1]
            )

        for i in range(start + 1, stop):
            if i + 1 < n:
                new_chromosome[i] = (
                    0.25 * chromosome[i - 1] +
                    0.5 * chromosome[i] +
                    0.25 * chromosome[i + 1]
                )

        if stop - 1 >= 0:
            new_chromosome[stop] = (
                0.67 * chromosome[stop] + 0.33 * chromosome[stop - 1]
            )

        individual.chromosome = new_chromosome
