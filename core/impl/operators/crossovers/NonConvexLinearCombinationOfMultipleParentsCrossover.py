import numpy as np
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation
from core import Individual, Population
from core.operators import Crossover
from core.fitness_function import FitnessFunction


class NonConvexLinearCombinationOfMultipleParentsCrossover(Crossover):
    """
    A class that implements Non-convex Linear Combination of Multiple Parents Crossover (Algorytmy genetyczne: kompendium. T. 1 p.243)

    :param how_many_individuals: Number of offspring to generate.
    :type how_many_individuals: int
    :param fitness_function: The fitness function used to evaluate individuals.
    :type fitness_function: FitnessFunction
    :param probability: Probability of applying the crossover.
    :type probability: float
    :param q: Number of parents to combine in the crossover (default: 8).
    :type q: int
    :param l: Lower bound for randomly generated weights (default: -0.3).
    :type l: float
    :param r: Upper bound for randomly generated weights (default: 1.3).
    :type r: float
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 0,
        q: int = 8,
        l: float = -0.3,
        r: float = 1.3,
    ):
        """ """
        super().__init__(how_many_individuals, probability)
        self.q = q
        self.l = l
        self.r = r
        self.variable_domains = fitness_function.variable_domains

    def _cross(self, population_parent: Population) -> Population:
        """
        Produces a single offspring using a non-convex linear combination
        of `q` randomly selected parents.

        Steps:
        1. Randomly selects `q` parents.
        2. Generates `q` random weights in the range [l, r] and normalizes them.
        3. Computes a weighted sum of the selected parents' chromosomes.

        :param population_parent: Population of parents.
        :type population_parent: Population
        :return: A new individual created as a weighted combination of parents.
        :rtype: Individual
        """
        while True:
            # if len(population_parent.population) < self.q:
            #    raise ValueError("Population must have at least 'q' individuals.")
            indices = np.random.choice(
                len(population_parent.population), size=self.q, replace=False
            )
            chosen_parents = [population_parent.population[i] for i in indices]

            while True:
                elements = np.random.uniform(self.l, self.r, self.q)
                sum_elements = np.sum(elements)
                if sum_elements != 0:
                    break
            elements /= sum_elements
            parent_chromosomes = np.array(
                [parent.chromosome for parent in chosen_parents]
            )
            weighted_chromosomes = parent_chromosomes.T * elements
            child_chromosome = np.sum(weighted_chromosomes, axis=1)
            low_bounds = np.array([low for low, _ in self.variable_domains])
            high_bounds = np.array([high for _, high in self.variable_domains])
            if np.all(
                (child_chromosome >= low_bounds) & (child_chromosome <= high_bounds)
            ):
                break
        return Population(population=[Individual(chromosome=child_chromosome)])
