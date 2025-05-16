import numpy as np
import random
from core import Individual, Population, Representation
from core.operators import Crossover


class FuzzyCrossover(Crossover):
    """
    Implements Fuzzy Crossover using bimodal triangular distributions. (Algorytmy genetyczne. Kompendium, t. 1, str. 211-212).

    :param how_many_individuals: Number of offspring to generate.
    :param probability: Probability of applying crossover.
    :param d: Spread parameter for fuzzy sets (default: 0.5).
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0, d: float = 0.5):
        super().__init__(how_many_individuals, probability)
        self.d = d

    def _cross(self, population_parent: Population) -> Population:
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace=False)

        parent_1_chromosome = np.array(parent_1.chromosome)
        parent_2_chromosome = np.array(parent_2.chromosome)

        offspring_genes = []

        for xi, yi in zip(parent_1_chromosome, parent_2_chromosome):
            if xi > yi:
                xi, yi = yi, xi

            range_width = abs(yi - xi)
            d_scaled = self.d * range_width

            if d_scaled == 0:
                alpha = xi
            else:
                domain_x = (xi - d_scaled, xi + d_scaled)
                domain_y = (yi - d_scaled, yi + d_scaled)

                if random.random() < 0.5:
                    alpha = np.random.triangular(domain_x[0], xi, domain_x[1])
                else:
                    alpha = np.random.triangular(domain_y[0], yi, domain_y[1])

            offspring_genes.append(alpha)

        offspring = Individual(chromosome=np.array(offspring_genes))
        return Population([offspring])
