import numpy as np
from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function import FitnessFunction

class SingleArithmeticalCrossover(Crossover):
    """
    Implements Single Arithmetical Crossover using a single crossover point.

    :param how_many_individuals: Number of offspring to generate.
    :param fitness_function: Fitness function that rovides lower and upper bounds for genes.
    :param probability: Probability of performing crossover.
    """
    
    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, fitness_function = FitnessFunction, probability: float = 0):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace=False)
        parent_1_chromosome = np.array(parent_1.chromosome)
        parent_2_chromosome = np.array(parent_2.chromosome)

        offspring_1 = []
        offspring_2 = []

        size = min(len(parent_1), len(parent_2))

        crossover_point = np.random.randint(0, size)

        x_lambda, y_lambda = parent_1_chromosome[crossover_point], parent_2_chromosome[crossover_point]
        x_lower, x_upper = self.fitness_function.variable_domains[crossover_point]

        if x_lambda > y_lambda:
            alpha_max = max((x_lower - y_lambda) / (x_lambda - y_lambda), (x_upper - x_lambda) / (y_lambda - x_lambda))
            alpha_min = min((x_lower - x_lambda) / (y_lambda - x_lambda), (x_upper - y_lambda) / (x_lambda - y_lambda))
        elif x_lambda < y_lambda:
            alpha_max = max((x_lower - x_lambda) / (y_lambda - x_lambda), (x_upper - y_lambda) / (x_lambda - y_lambda))
            alpha_min = min((x_lower - y_lambda) / (x_lambda - y_lambda), (x_upper - x_lambda) / (y_lambda - x_lambda))
        elif x_lambda == y_lambda:
            alpha_min = alpha_max = 0
        
        alpha = np.random.uniform(alpha_min, alpha_max)

        for i in range(crossover_point - 1):
            offspring_1.append(parent_1_chromosome[i])
            offspring_2.append(parent_2_chromosome[i]) 

        offspring_1.append(alpha * y_lambda + (1 - alpha) * x_lambda)
        offspring_2.append(alpha * x_lambda + (1 - alpha) * y_lambda)

        for i in range(crossover_point + 1, size):
            offspring_1.append(parent_2_chromosome[i])
            offspring_2.append(parent_1_chromosome[i])

        return Population(
            population=[
                Individual(chromosome=offspring_1, value=0),
                Individual(chromosome=offspring_2, value=0),
            ]
        )