import numpy as np
from Population import Population
from Mutation import Mutation
from Individual import Individual
import random
import math


class SphereMutation(Mutation):
    def __init__(self, probability: float):
        """
        Constructor for the Sphere Mutation class.

        Parameters:
            probability (float): Probability of mutation (between 0 and 1).
        """
        super().__init__(probability)

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Applies Sphere Mutation to the individual's chromosome.

        Parameters:
            individual (Individual): A single individual to mutate.
            population (Population): The population containing individuals.

        Returns:
            None: The population with mutated individuals.
        """
        self.sphere_mutation(individual.chromosome)

    def sphere_mutation(self, chromosome):
        # Use the mutation probability from the Mutation superclass
        if np.random.rand() <= self.probability:

            k, q = random.sample(range(len(chromosome)), 2)
            a = np.random.uniform(0, 1)
            B = math.sqrt((chromosome[k] / chromosome[q]) ** 2 * (1 - a ** 2) + 1)

            chromosome[k] = a * chromosome[k]
            chromosome[q] = B * chromosome[q]

        return None