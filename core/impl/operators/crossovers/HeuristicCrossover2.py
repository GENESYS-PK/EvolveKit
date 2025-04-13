import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function.FitnessFunction import FitnessFunction


class HeuristicCrossover2(Crossover):
    """
    Implements Heuristic Crossover 2 for optimization problems (minimization or maximization).
    Ensures offspring genes stay within specified per-gene ranges.

    :param how_many_individuals: Number of individuals to create in the offspring.
    :param fitness_function: Fitness function used to evaluate individuals.
    :param probability: Probability of performing the crossover.
    :param maximize: Whether the optimization problem is a maximization (True) or minimization (False).
    :param ranges: List of (min, max) tuples for each gene.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 0,
        maximize: bool = False,
        ranges: list[tuple[float, float]] = None
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function
        self.maximize = maximize
        self.ranges = ranges

    def _cross(self, population_parent: Population) -> Population:
        offspring_population = []

        for _ in range(self.how_many_individuals):
            parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace=False)

            X = np.array(parent_1.chromosome)
            Y = np.array(parent_2.chromosome)

            f_X = self.fitness_function.calculate_individual_value(parent_1)
            f_Y = self.fitness_function.calculate_individual_value(parent_2)

            alpha = np.random.uniform(0, 1)

            if (self.maximize and f_X <= f_Y) or (not self.maximize and f_X >= f_Y):
                offspring_chromosome = alpha * (Y - X) + Y
            else:
                offspring_chromosome = alpha * (X - Y) + X

            if self.ranges:
                clipped_genes = [
                    np.clip(gene, r_min, r_max)
                    for gene, (r_min, r_max) in zip(offspring_chromosome, self.ranges)
                ]
                offspring_chromosome = np.array(clipped_genes)

            offspring = Individual(chromosome=offspring_chromosome)
            offspring_population.append(offspring)

        return Population(offspring_population)
