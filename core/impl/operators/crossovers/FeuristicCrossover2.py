import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function.FitnessFunction import FitnessFunction

class FeuristicCrossover2(Crossover):
    """
    Implements Heuristic Crossover 2 for minimization problems (Algorytmy genetyczne. Kompendium, t. 1, str. 194-195).

    :param how_many_individuals: Number of individuals to create in the offspring.
    :param fitness_function: Fitness function used to evaluate individuals.
    :param probability: Probability of performing the crossover.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 0
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace=False)

        X = np.array(parent_1.chromosome)
        Y = np.array(parent_2.chromosome)

        f_X = self.fitness_function.calculate_individual_value(parent_1)
        f_Y = self.fitness_function.calculate_individual_value(parent_2)

        alpha = np.random.uniform(0, 1)

        if f_X >= f_Y:
            offspring_chromosome = alpha * (Y - X) + Y
        else:
            offspring_chromosome = alpha * (X - Y) + X

        offspring = Individual(chromosome=offspring_chromosome)
        return Population([offspring])
