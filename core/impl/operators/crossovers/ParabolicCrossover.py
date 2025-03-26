import numpy as np
from scipy.optimize import fsolve

from core import Individual, Population, Representation
from core.operators import Crossover
from core.fitness_function import FitnessFunction


class ParabolicCrossover(Crossover):
    """
    Implements Parabolic Crossover, where the offspring is the vertex of a paraboloid
    passing through (n+1) parents in R^n space. (Algorytmy genetyczne. Kompendium, t. 1, str. 249-251).
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, fitness_function: FitnessFunction, probability: float = 0):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        chromosome_length = len(population_parent.population[0])
        num_parents = chromosome_length + 1
        parents = np.random.choice(population_parent.population, num_parents, replace=False)

        chromosomes = [np.array(p.chromosome) for p in parents]
        fitnesses = [self.fitness_function.calculate_individual_value(p) for p in parents]

        def get_equation(chromosome, fitness_val):
            def equation(v):
                return sum((chromosome[i] - v[i]) ** 2 for i in range(len(v))) + \
                       self.fitness_function.calculate(v) - fitness_val
            return equation

        equations = [get_equation(chromosomes[i], fitnesses[i]) for i in range(num_parents)]

        def system(v):
            eq_vals = [eq(v) for eq in equations]
            return [eq_vals[i] - eq_vals[i + 1] for i in range(len(eq_vals) - 1)]

        initial_guess = [0.0] * chromosome_length
        root = fsolve(system, initial_guess)

        offspring = Individual(chromosome=np.array(root))
        return Population([offspring])
