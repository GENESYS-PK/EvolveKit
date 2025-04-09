import numpy as np
from core import Individual, Population
from core.operators import Mutation
from core.Representation import Representation

class DynamicMutationE(Mutation):
    """
    Version E – swap mutation. (Algorytmy genetyczne. Kompendium, t. 2, str. 199 - 200).

    This mutation randomly selects a mutation index λ and swaps the value at λ
    with the value at λ + 1. This is useful for shifting or reordering values slightly.

    :param probability: Probability of applying the mutation to an individual.
    :returns: None – the individual's chromosome is mutated in place.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float = 1.0):
        super().__init__(probability)

    def _mutate(self, individual: Individual, population: Population) -> None:
        if np.random.uniform(0, 1) > self.probability:
            return

        chromosome = individual.chromosome.copy()
        n = len(chromosome)

        if n < 2:
            return  

        lam = np.random.randint(0, n - 1)

        chromosome[lam], chromosome[lam + 1] = chromosome[lam + 1], chromosome[lam]

        individual.chromosome = chromosome
