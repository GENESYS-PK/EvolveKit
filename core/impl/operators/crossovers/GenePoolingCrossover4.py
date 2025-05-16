import numpy as np
from core import Population
from core.operators import Crossover
from core.Population import Population


class GenePoolingCrossover4(Crossover):
    def __init__(
        self,
        how_many_individuals: int,
        allow_gene_duplicates: bool = True,
        minimize: bool = True,
        probability: float = 0,
    ):
        """
        Constructor for the GenePoolingCrossover4 (GPX-4) class.

        :param how_many_individuals: The number of individuals to create in the offspring.
        :param probability: The probability of performing the crossover operation.
        :param minimize: Whether the crossover is for a minimization problem (default True)
                         or maximization (False).
        :param allow_gene_duplicates: Whether we allow to generate an offspring with gene duplicates (default True)
                                      or throw an exception (False).
        """
        super().__init__(how_many_individuals, probability)
        self.allow_gene_duplicates = allow_gene_duplicates
        self.minimize = minimize

    def _cross(self, population_parent: Population) -> Population:
        """
        Perform Gene-Pooling Crossover 4 on the parent population.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        :raises ValueError: If the population size is less than 2.
        :raises Exception: If we didn't allow to generate an offspring with gene duplicates.
        """
        offspring_population = Population()

        # Randomly select two parents from the parent population
        parent_indices = np.random.choice(
            population_parent.population_size, 2, replace=False
        )
        parent_x = population_parent.population[parent_indices[0]].chromosome
        parent_y = population_parent.population[parent_indices[1]].chromosome

        # Finding the sorted union of the two individuals (from the smallest to the largest - minimization)
        if self.minimize:
            S = np.union1d(parent_x, parent_y)
        else:
            S = np.flip(np.union1d(parent_x, parent_y))

        # Finding the size of the new individual
        offspring_individual_size = np.random.randint(
            2, (len(parent_x) + len(parent_y)) + 1
        )

        # We can either throw an exception
        if not self.allow_gene_duplicates and len(S) < offspring_individual_size:
            raise Exception("We can't generate an offspring with gene duplicates.")

        # or return an offspring which would contain duplicates.
        if len(S) < offspring_individual_size:
            S = np.concatenate((parent_x, parent_y))
            S.sort()
            if not self.minimize:
                S = np.flip(S)

        # Selecting the new individual
        offspring_individual = S[:offspring_individual_size]

        # Add new individual to the offspring population
        offspring_population.add_to_population(offspring_individual)
        return offspring_population
