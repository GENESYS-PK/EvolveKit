import numpy as np
from core.Mutation import Mutation
from core.Individual import Individual
from core.Population import Population


class DebGoyalMutation(Mutation):
    def __init__(self, n: int, delta_max: float, probability: float = 0):
        """
        Constructor for the Deb & Goyal Mutation (DGM) class.

        :param probability: Probability of mutation (float between 0 and 1).
        :param n: Distribution index for mutation.
        :param delta_max: Maximum allowed perturbation.
        :raises ValueError: If the distribution index is less than 1.
        """
        super().__init__(probability)
        if int(n) < 1:
            raise ValueError("The distribution index must be at least 1 to perform the DebGoyalMutation operation.")
        self.n = n
        self.delta_max = delta_max

    def mutate(self, population_parent: Population) -> Population:
        """
        Perform the Deb & Goyal Mutation operations for the entire population.

        :param population_parent: The population to perform the mutation operation on.
        :returns: The mutated population.
        """
        for individual in population_parent.population:
            self._mutate(individual, population_parent)

        return population_parent

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Applies Deb & Goyal mutation to the individual's chromosome.

        :param individual: A single individual to mutate.
        :param population: The population containing individuals.
        :returns: None.
        """
        chromosome = individual.chromosome
        if self.n > len(chromosome):
            self.n = len(chromosome)

        for i in range(self.n):
            if np.random.uniform(0, 1) <= self.probability:
                u = np.random.uniform(0, 1)

                if u < 0.5:
                    delta = np.power(2 * u, 1 / (self.n + 1)) - 1
                else:
                    delta = 1 - np.power(2 * (1 - u), 1 / (self.n + 1))

                chromosome[i] += delta * self.delta_max

        individual.chromosome = chromosome
