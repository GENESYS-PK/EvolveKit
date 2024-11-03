import numpy as np

from core import Population
from core.operators import Crossover
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation


class BlendCrossoverAlfa(Crossover):
    """
    A class that implements Blend Crossover α - BLX-α.

    :param how_many_individuals: The number of individuals to create in the offspring.
    :param probability: The probability of performing the crossover operation.
    :param alfa: The parameter used for blending two genes, must be positive, preferably closer to 0 and less than 1.
    :raises ValueError: If the alfa parameter is not positive or population size is less than 2.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(
        self, how_many_individuals: int, probability: float = 0, alfa: float = 0.1
    ):
        super().__init__(how_many_individuals, probability)
        if float(alfa) <= 0:
            raise ValueError("The alfa parameter must be positive")
        self.alfa = alfa

    def _cross(self, population_parent: Population) -> Population:
        """
        :param population_parent: The population to perform the crossover operation on.
        :returns: The offspring population (always 2 children).
        """
        individual_index1 = np.random.randint(population_parent.population_size)
        individual_index2 = np.random.randint(population_parent.population_size)

        while individual_index1 == individual_index2:
            individual_index2 = np.random.randint(population_parent.population_size)

        def blend_one_pair_of_genes(g1: float, g2: float) -> float:
            delta = abs(g1 - g2)
            alfa_delta = self.alfa * delta
            return np.random.uniform(min(g1, g2) - alfa_delta, max(g1, g2) + alfa_delta)

        # lambda, because iterators can be used only once, so it will be used to return a new iterator each time
        tmp_zip = lambda: zip(
            population_parent.population[individual_index1].chromosome,
            population_parent.population[individual_index2].chromosome,
        )

        child1 = Individual(
            chromosome=np.array(
                [blend_one_pair_of_genes(g1, g2) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )
        child2 = Individual(
            chromosome=np.array(
                [blend_one_pair_of_genes(g1, g2) for g1, g2 in tmp_zip()], dtype=float
            ),
            value=0,
        )

        return Population(population=[child1, child2])
