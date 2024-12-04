import numpy as np
from core.Individual import Individual
from core.Population import Population
from core.Representation import Representation
from core import Individual, Population
from core.operators import Crossover


class NonConvexLinearCombinationOfMultipleParentsCrossover(Crossover):
    """ """

    allowed_representation = [Representation.REAL]

    def __init__(
        self,
        how_many_individuals: int,
        probability: float = 0,
        q: int = 8,
        l: float = -0.3,
        r: float = 1.3,
    ):
        """ """
        super().__init__(how_many_individuals, probability)
        self.q = q
        self.l = l
        self.r = r

    def _cross(self, population_parent: Population) -> Population:
        """ """

        chosen_parents = np.random.choice(population_parent, size=self.q, replace=False)
        elements = np.random.uniform(self.l, self.r, self.q)
        sum_elements = np.sum(elements)
        elements = elements / sum_elements
        chromosome_size = len(chosen_parents[0].chromosome)
        child_chromosome = np.zeros(chromosome_size)
        for parent, beta in zip(chosen_parents, elements):
            child_chromosome += parent.chromosome * beta
        return Individual(chromosome=child_chromosome)
