import numpy as np
from core import Population, Individual
from core.operators import Crossover
from core.Population import Population
from core.fitness_function import FitnessFunction
from core.Representation import Representation


class MutualFitnessCrossover(Crossover):
    """
    Implements Mutual Fitness Crossover based on the algorithm described.
    
    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fitness_function: A function that is being optimized.
    :param probability: The probability of performing the crossover operation.
    :param max_generations: The maximum number of generations in the evolutionary process.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 0.7,
        max_generations: int = 100,
        current_generation: int=0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function
        self.max_generations = max_generations
        self.generation = current_generation

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population.
        """
        new_population = []
        pop_size = len(population_parent.population)

        for i in range(pop_size):
            X = population_parent.population[i]

            Y = max(
                (ind for ind in population_parent.population if ind != X),
                key=lambda ind: (self.fitness_function.calculate_individual_value(X) - self.fitness_function.calculate_individual_value(ind)) ** 2 /
                               np.linalg.norm(X.chromosome - ind.chromosome) ** 2,
            )

            v_t_M = 0.75 * (self.generation / self.max_generations) + 0.25
            alpha_lower_bound = 1 - 0.2 * v_t_M
            alpha_upper_bound = 1 + v_t_M

            offspring_chromosome = np.copy(X.chromosome)
            
            alpha = np.random.uniform(alpha_lower_bound, alpha_upper_bound)

            if self.fitness_function.calculate_individual_value(X) >= self.fitness_function.calculate_individual_value(Y):
                offspring_chromosome = alpha * X.chromosome + (1 - alpha) * Y.chromosome
            else:
                offspring_chromosome = alpha * Y.chromosome + (1 - alpha) * X.chromosome

            offspring = Individual(chromosome=offspring_chromosome)
            new_population.append(offspring)
        
        self.generation += 1
        
        return Population(new_population)
