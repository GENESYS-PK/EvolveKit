import numpy as np
from core import Individual, Population
from core.operators import Mutation
from core.Representation import Representation

class DynamicMutationA(Mutation):
    """
    Version A – directional mutation with exponential decay. (Algorytmy genetyczne. Kompendium, t. 2, str. 197-198).

    This mutation selects one gene and mutates it in a direction (increase or decrease)
    using an exponential decay function for the step size, depending on a normally
    distributed alpha value.

    :param lower_bounds: A list or array of lower bounds for each gene.
    :param upper_bounds: A list or array of upper bounds for each gene.
    :param probability: Probability of applying mutation to an individual (default: 0.1).
    :returns: None – the individual's chromosome is mutated in place.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, lower_bounds, upper_bounds, probability: float = 0.1):
        super().__init__(probability)
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

    def _mutate(self, individual: Individual, population: Population) -> None:
        chromosome = individual.chromosome.copy()
        n_genes = len(chromosome)

        if np.random.uniform(0, 1) > self.probability:
            return

        mutation_index = np.random.randint(0, n_genes)

        alpha = np.random.normal(0, 1)

        rnd = np.random.uniform(0, 1)

        x = chromosome[mutation_index]
        x_l = self.lower_bounds[mutation_index]
        x_u = self.upper_bounds[mutation_index]

        exp_component = 1 - np.exp(-abs(alpha))

        if rnd < 0.5:
            x_new = x + (x_u - x) * exp_component
        else:
            x_new = x - (x - x_l) * exp_component

        chromosome[mutation_index] = np.clip(x_new, x_l, x_u)
        individual.chromosome = chromosome
