import numpy as np
from core import Individual, Population
from core.operators import Mutation
from core.Representation import Representation

class DynamicMutationB(Mutation):
    """
    Version B – creep mutation with boundary correction. (Algorytmy genetyczne. Kompendium, t. 2, str. 198).

    Each gene is modified slightly by adding a small, normally distributed value scaled by β.
    If the new value exceeds the boundaries, it is corrected iteratively by halving the
    mutation step until it falls within the allowed range.

    :param lower_bounds: A list or array of lower bounds for each gene.
    :param upper_bounds: A list or array of upper bounds for each gene.
    :param probability: Probability of applying mutation to an individual (default: 1.0).
    :param beta: Scaling factor that controls mutation intensity (default: 0.02).
    :returns: None – the individual's chromosome is mutated in place.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, lower_bounds, upper_bounds, probability: float = 1.0, beta: float = 0.02):
        super().__init__(probability)
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.beta = beta

    def _mutate(self, individual: Individual, population: Population) -> None:
        chromosome = individual.chromosome.copy()
        n_genes = len(chromosome)

        for i in range(n_genes):
            alpha = np.random.normal(0, 1)
            x = chromosome[i]
            x_l = self.lower_bounds[i]
            x_u = self.upper_bounds[i]

            delta = alpha * self.beta * (x_u - x_l)
            x_new = x + delta
            gamma = abs(delta)

            while x_new > x_u or x_new < x_l:
                gamma /= 2
                if x_new > x_u:
                    x_new = x + gamma
                elif x_new < x_l:
                    x_new = x - gamma

            chromosome[i] = x_new

        individual.chromosome = chromosome
