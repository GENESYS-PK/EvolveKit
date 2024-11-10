import numpy as np
from core import Population
from core.operators import Mutation
from core.Individual import Individual
from core.Representation import Representation


class BoundaryMutation(Mutation):
    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float, pbm: float, boundaries: list):
        """
        Constructor for the BoundaryMutation class.

        Parameters:
            probability (float): Probability of mutation (between 0 and 1).
            pbm (float): The probability of mutating each gene within an individual.
            boundaries (list): The allowed boundaries for each gene.
        """
        super().__init__(probability)
        self.pbm = pbm  # Probability of boundary mutation for each gene
        self.boundaries = boundaries  # List of [min, max] boundaries for each gene

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Applies Boundary Mutation to the individual's chromosome.

        Parameters:
            individual (Individual): A single individual to mutate.
            population (Population): The population containing individuals.

        Returns:
            None: The population with mutated individuals.
        """
        self.boundary_mutation(individual.chromosome)

    def boundary_mutation(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.uniform(0, 1) <= self.pbm:
                # Mutate gene to either lower or upper boundary
                if np.random.uniform(0, 1) < 0.5:
                    chromosome[i] = self.boundaries[i][0]
                else:
                    chromosome[i] = self.boundaries[i][1]

        return None
