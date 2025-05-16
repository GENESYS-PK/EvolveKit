import numpy as np
from core import Population, Individual
from core.operators import Crossover
from core.Representation import Representation
from core.fitness_function import FitnessFunction


class GuidedCrossover(Crossover):
    """
    Implements Mutual Fitness Crossover based on the EGAX algorithm.

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
        current_generation: int = 0,
    ):
        super().__init__(how_many_individuals, probability)

        if not hasattr(fitness_function, "calculate_individual_value"):
            raise TypeError(
                f"Expected fitness_function to have 'calculate_individual_value', got {type(fitness_function)}"
            )

        self.fitness_function = fitness_function
        self.max_generations = max_generations
        self.generation = current_generation

    def _cross(self, population_parent: Population) -> Population:
        new_population = []
        pop_size = len(population_parent.population)

        for i in range(pop_size):
            X = population_parent.population[i]
            X_chromosome = np.array(X.chromosome)

            def fitness_diff(ind):
                Y_chromosome = np.array(ind.chromosome)
                f_X = self.fitness_function.calculate_individual_value(X)
                f_Y = self.fitness_function.calculate_individual_value(ind)
                numerator = (f_X - f_Y) ** 2
                denominator = np.linalg.norm(X_chromosome - Y_chromosome) ** 2 + 1e-8
                return numerator / denominator

            potential_partners = [ind for ind in population_parent.population if ind is not X]
            Y = max(potential_partners, key=fitness_diff)

            Y_chromosome = np.array(Y.chromosome)

            v_t_M = 0.75 * (self.generation / self.max_generations) + 0.25
            alpha_lower_bound = 1 - 0.2 * v_t_M
            alpha_upper_bound = 1 + v_t_M
            alpha = np.random.uniform(alpha_lower_bound, alpha_upper_bound)

            f_X = self.fitness_function.calculate_individual_value(X)
            f_Y = self.fitness_function.calculate_individual_value(Y)

            if f_X >= f_Y:
                offspring_chromosome = alpha * X_chromosome + (1 - alpha) * Y_chromosome
            else:
                offspring_chromosome = alpha * Y_chromosome + (1 - alpha) * X_chromosome

            domains = np.array(self.fitness_function.variable_domains)
            offspring_chromosome = np.minimum(
            np.maximum(offspring_chromosome, domains[:, 0]),
            domains[:, 1]
            )

            new_population.append(Individual(chromosome=offspring_chromosome))

        self.generation += 1
        return Population(new_population)

