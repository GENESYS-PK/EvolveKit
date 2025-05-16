import numpy as np
from typing import List
from scipy.optimize import fsolve

from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function.FitnessFunction import FitnessFunction


class ParabolicCrossover(Crossover):
    """
    Implements Parabolic Crossover, where the offspring is the vertex of a paraboloid
    passing through (n+1) parents in R^n space. (Algorytmy genetyczne. Kompendium, t. 1, str. 249-251).

    :param how_many_individuals: Number of offspring to generate.
    :param fitness_function: Fitness function to evaluate fitness.
    :param probability: Probability of crossover application.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 1.0
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        offspring_population = []
        chromosome_length = len(population_parent.population[0].chromosome)

        def get_equation(chromosome: List[float]):
            def equation(vector: List[float]):
                diff_sum = sum((chromosome[i] - vector[i]) ** 2 for i in range(chromosome_length))
                individual_vector = Individual(chromosome=np.array(vector))
                individual_chromosome = Individual(chromosome=np.array(chromosome))

                return (
                    diff_sum
                    + self.fitness_function.calculate_individual_value(individual_vector)
                    - self.fitness_function.calculate_individual_value(individual_chromosome)
                )
            return equation

        for _ in range(self.how_many_individuals):
            try:
                parents = np.random.choice(population_parent.population, chromosome_length + 1, replace=False)
                chromosomes = [parent.chromosome.tolist() for parent in parents]

                system_funcs = [get_equation(ch) for ch in chromosomes]

                def system(vector: List[float]):
                    results = [f(vector) for f in system_funcs]
                    return [results[i] - results[i + 1] for i in range(len(results) - 1)]

                init = np.mean(chromosomes, axis=0)

                solution = fsolve(system, init)

                if not np.all(np.isfinite(solution)):
                    continue

                offspring = Individual(chromosome=np.array(solution, dtype=float))
                offspring_population.append(offspring)

            except Exception:
                continue

        return Population(offspring_population)
