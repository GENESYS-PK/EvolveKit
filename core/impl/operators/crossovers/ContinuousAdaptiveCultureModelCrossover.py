import numpy as np
from core import Population
from core.operators import Crossover
from core.Population import Population
from core.Representation import Representation


class ContinuousAdaptiveCultureModelCrossover(Crossover):
    """
    Implements Continuous Adaptive Culture Model Crossover method which creates offspring based on the environment from one parent.

    Algorytmy Genetyczne kompedium tom I Operator krzyżowania dla problemów numerycznych. Tomasz Gwiazda pp. 365-366.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param scaling_constant: Aa constant that controls the "distance" of the offspring from the parent.
                            Increasing the value of this constant increases the probability that the
                            offspring will be significantly "distanced" from the parent.
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self, how_many_individuals: int, scaling_constant: float, probability: float = 0
    ):
        super().__init__(how_many_individuals, probability)
        self.scaling_constant = scaling_constant

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring.
        """

        parent_1, parent_2 = np.random.choice(population_parent.population, 2)

        probability_of_parent_selection = np.random.uniform(0, 1)
        offspring = np.array()

        if probability_of_parent_selection < 0.5:
            alphas = np.random.uniform(-0.5, 0.5, size=parent_1.chromosome.shape)
            np.append(offspring, parent_1.chromosome + self.scaling_constant * alphas)
        else:
            alphas = np.random.uniform(-0.5, 0.5, size=parent_2.chromosome.shape)
            np.append(offspring, parent_2.chromosome + self.scaling_constant * alphas)

        return Population(population=[offspring])
