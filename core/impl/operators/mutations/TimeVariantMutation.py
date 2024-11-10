import numpy as np
from core import Population
from core.operators import Mutation
from core.Individual import Individual
from core.Representation import Representation


class TimeVariantMutation(Mutation):
    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float, ypsilon: float, t: int, M: int, k: int):
        """
        Constructor for the Time Variant Mutation class.

        Parameters:
            probability (float): Probability of mutation (between 0 and 1).
            ypsilon (float): A constant that controls the strength of the mutation.
            t: Current generation.
            M: Maximum number of generations.
            k: The index of the gene to mutate.
        """
        super().__init__(probability)
        self.ypsilon = ypsilon  # Controls mutation strength
        self.t = t  # Current generation
        self.M = M  # Maximum generations
        self.k = k  # Index of gene to mutate

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Applies Time Variant Mutation to the individual's chromosome.

        Parameters:
            individual (Individual): A single individual to mutate.
            population (Population): The population containing individuals.

        Returns:
            None: The population with mutated individuals.
        """
        self.time_variant_mutation(individual.chromosome)

    def time_variant_mutation(self, chromosome):
        for j in range(len(chromosome)):
            r = np.random.uniform(0, 1)
            sigma = 1 - (r ** (1 - self.t / self.M)) ** self.ypsilon
            a = np.random.normal(0, sigma**2)
            chromosome[self.k] = chromosome[self.k] + a

        return None
