import numpy as np

from core import Individual
from core.operators.Mutation import Mutation
from core.Population import Population
from core.Representation import Representation


class DFPMMutation(Mutation):
    """
    The class implements the Differential Fixed Point Mutation (DFPM), which is an extension
    of the non-uniform mutation method. In DFPM, two individuals are randomly selected, and
    a single gene is mutated based on the directional difference between them. Two offspring
    are generated using asymmetrical and symmetrical changes, and the best individual among
    the parent and both offspring is chosen.

    Based on: Tomasz Gwiazda – Genetic Algorithms Compendium Vol. II, Differential Mutation Operator,
    pp. 226–227.

    :param probability: The probability of applying the mutation.
    :param fitness_function: A callable used to evaluate individual fitness.
    :param minimize: Boolean flag indicating whether the problem is a minimization (True) or maximization (False).
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, fitness_function: callable, minimize: bool = True, probability: float = 1.0):
        super().__init__(probability)
        self.fitness_function = fitness_function
        self.minimize = minimize

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Applies the DFPM mutation operator to the provided individual.

        :param individual: The individual on which mutation is applied.
        :param population: The full population, used to select the second parent.
        :return: None. The individual is modified in-place.
        """
        chromosome_length = len(individual.chromosome)
        population_size = len(population.population)
        t = np.random.randint(1, 1000)

        while True:
            parent2 = population.population[np.random.randint(0, population_size)]
            if parent2 is not individual:
                break

        k = np.random.randint(0, chromosome_length)

        x_jk = individual.chromosome[k]
        x_uk = parent2.chromosome[k]

        b = 5.0
        r = np.random.rand()
        if np.random.rand() <= 0.5:
            delta = (x_uk - x_jk) * (1 - r ** ((1 - t / 1000.0) ** b))
            new_val = x_jk + delta
        else:
            delta = (x_jk - x_uk) * (1 - r ** ((1 - t / 1000.0) ** b))
            new_val = x_jk - delta

        child1 = np.copy(individual.chromosome)
        child1[k] = new_val

        child2 = np.copy(parent2.chromosome)
        child2[k] = x_uk + x_jk - new_val

        candidates = [individual.chromosome, child1, child2]
        fitnesses = [self.fitness_function(c) for c in candidates]

        best_idx = np.argmin(fitnesses) if self.minimize else np.argmax(fitnesses)
        individual.chromosome = np.copy(candidates[best_idx])