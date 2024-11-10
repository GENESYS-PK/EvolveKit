import numpy as np
from core import Population
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class SingleGenCrossover(Crossover):
    """
    Implements SingleGenCrossover

    Algorytmy Genetyczne kompedium tom I Operator krzyżowania dla problemów numerycznych. Tomasz Gwiazda pp. 339-340.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 0):
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring.
        """

        population_size = len(population_parent.population)

        chromosome_size = population_parent.population[0].chromosome.size
        lambda_index = np.random.randint(1, chromosome_size)

        offsprings = []

        for j in range(population_size):
            new_chromosome = []
            for i in range(1, chromosome_size + 1):
                if i != lambda_index:
                    new_chromosome.append(
                        population_parent.population[j].chromosome[i - 1]
                    )
                else:
                    if j + 1 <= population_size / 2:
                        alpha = np.random.random()
                        l = j + int(population_size / 2)
                        new_chromosome.append(
                            (1 - alpha)
                            * population_parent.population[j].chromosome[i - 1]
                            + alpha * population_parent.population[l].chromosome[i - 1]
                        )
                    else:
                        alpha = np.random.random()
                        l = j - int(population_size / 2)
                        new_chromosome.append(
                            (1 - alpha)
                            * population_parent.population[j].chromosome[i - 1]
                            + alpha * population_parent.population[l].chromosome[i - 1]
                        )

            offsprings.append(
                Individual(chromosome=new_chromosome, value=0)
            )  # nie wiem dałam tu 0 bo było u kogoś innego tez tak, jeżeli tu powinno iść wywołanie fittness function proszę o info

        return Population(population=offsprings)
