import numpy as np

from core import Population, Individual
from core.operators import Crossover
from core.Population import Population
from core.fitness_function import FitnessFunction
from core.Representation import Representation


class LinearCrossover(Crossover):
    """
    Implements Linear Crossover.

     Algorytmy Genetyczne kompedium tom I Operator krzyżowania dla problemów numerycznych. Tomasz Gwiazda pp. 193.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param fittness_function: A function that is being optimized.
    :param probability: The probability of performing the crossover operation.
    :param problem_of_maximisation: The variable indicating if in algorithm we are looking for maximum or minimum of function.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self,
        how_many_individuals: int,
        fittness_function: FitnessFunction,
        probability: float = 0,
        problem_of_maximisation: bool = True
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fittness_function
        self.problem_of_maximisation = problem_of_maximisation

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring.
        """
        parent_1, parent_2 = np.random.choice(population_parent.population, 2, replace = False)

        if not isinstance(parent_1.chromosome, np.ndarray) or not isinstance(parent_2.chromosome, np.ndarray):

            parent_1_chromosome = np.array(parent_1.chromosome)
            parent_2_chromosome = np.array(parent_2.chromosome)
        else:
            parent_1_chromosome = parent_1.chromosome
            parent_2_chromosome = parent_2.chromosome

        z = (1/2) * parent_1_chromosome + (1/2) * parent_2_chromosome
        v = (3/2) * parent_1_chromosome - (1/2) * parent_2_chromosome
        w = (3/2) * parent_2_chromosome - (1/2) * parent_1_chromosome

        offspring_1 = Individual(z)
        offspring_2 = Individual(v)
        offspring_3 = Individual(w)

        newPopulation = [offspring_1, offspring_2, offspring_3]
        fitness_values = [self.fitness_function.calculate_individual_value(offspring) for offspring in newPopulation]

        if self.problem_of_maximisation:
            min_index = fitness_values.index(min(fitness_values))
            newPopulation.pop(min_index)
        else:
            max_index = fitness_values.index(max(fitness_values))
            newPopulation.pop(max_index)


        return Population(newPopulation)






