import numpy as np
from core import Mutation, Individual, Population
from core.Representation import Representation


class BreederGAMutation(Mutation):
    """
    Implements Breeder GA mutation where gene values are adjusted based on mutation ranges.

    Algorytmy genetyczne : kompendium. T. 2, Operator mutacji dla problemów numerycznych. / T. D. Gwiazda, 2021 (strona 206)

    :param probability: Probability of mutating each gene in an individual.
    :param k: Parameter controlling mutation precision.
    :param ranges: List of mutation ranges for each gene.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, k: float, ranges: list, probability: float = 0):
        super().__init__(probability)
        self.k = k
        self.ranges = np.array(ranges)

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Apply Breeder GA mutation to an individual's chromosome.

        :param individual: The individual to mutate.
        :param population: The population containing the individual.
        :returns: None
        """
        chromosome = individual.chromosome

        for i in range(len(chromosome)):
            if np.random.uniform(0, 1) <= self.probability:
                alfa = np.random.uniform(0, 1)
                delta = self.ranges[i] * 2 ** (-self.k * alfa)

                if np.random.uniform(0, 1) <= 0.5:
                    chromosome[i] -= delta
                else:
                    chromosome[i] += delta
