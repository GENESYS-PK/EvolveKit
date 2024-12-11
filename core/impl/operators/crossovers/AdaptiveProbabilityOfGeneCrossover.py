import numpy as np
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation
from core.fitness_function.FitnessFunction import FitnessFunction


class AdaptiveProbabilityOfGeneCrossover(Crossover):
    """
    A class that implements Adaptive Probability of Gene Crossover (Algorytmy genetyczne: kompendium. T. 1 p. 345)


    :param how_many_individuals: Number of individuals in the population.
    :type how_many_individuals: int
    :param fitness_function: Fitness function used to evaluate the parents.
    :type fitness_function: FitnessFunction
    :param probability: Probability of performing crossover.
    :type probability: float
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        fitness_function: FitnessFunction,
        probability: float = 0,
    ):
        super().__init__(how_many_individuals, probability)
        self.fitness_function = fitness_function

    def _cross(self, population_parent: Population) -> Population:
        """
        Steps:
        1. Randomly select two parents from the population.
        2. Calculate the fitness of both parents.
        3. Compute the probability of selecting genes from the first parent based on fitness.
        4. For each gene, randomly choose it from either parent based on the calculated probability.
        5. Create the offspring with the selected genes.

        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring.
        """
        fitness_x, fitness_y = 0
        while fitness_x + fitness_y == 0:
            x_parent, y_parent = np.random.choice(
                population_parent.population, size=2, replace=False
            )
            fitness_x = self.fitness_function.fitness_function(x_parent.chromosome)
            fitness_y = self.fitness_function.fitness_function(y_parent.chromosome)

        probability_of_x = fitness_y / (fitness_x + fitness_y)
        random_values = np.random.rand(len(x_parent.chromosome))
        offspring_chromosome = np.where(
            random_values <= probability_of_x, x_parent.chromosome, y_parent.chromosome
        )
        offspring = Individual(chromosome=offspring_chromosome)
        return Population(population=[offspring])
