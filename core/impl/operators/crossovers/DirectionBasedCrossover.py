import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function import FitnessFunction

class DirectionBasedCrossover(Crossover):
    """
    Implements Direction-based crossover algorithm.
    
    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fitness_function: Function used to calculate value of indiviedual.
    :param probability: Probability of performing crossover.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, fitness_function: FitnessFunction, probability: float = 0):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace=False)

        f_x = self.fitness_function.calculate_individual_value(parent_1)
        f_y = self.fitness_function.calculate_individual_value(parent_2)

        if f_x < f_y:
            parent_1, parent_2 = parent_2, parent_1

        size = min(len(parent_1), len(parent_2))
        offspring_chromosome = []

        psd = (f_x - f_y) / f_x
        alpha = np.random.uniform(0, 1)

        for i in range(size):
            new_xi = parent_2[i] + alpha * (parent_1[i] - parent_2[i]) * psd
            offspring_chromosome.append(new_xi)

        offspring = Individual(chromosome=offspring_chromosome, value=0)

        return Population(population=[offspring])