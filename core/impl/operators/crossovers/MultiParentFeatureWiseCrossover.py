import numpy as np

from BlendCrossoverAlfa import BlendCrossoverAlfa
from core import Individual
from core.Population import Population
from core.Representation import Representation
from core.operators.Crossover import Crossover


class MultiParentFeatureWiseCrossover(Crossover):
    """
    A class that implements Multi Parent Feature Wise Crossover.
    Internally it uses the BlendCrossoverAlfa to blend two genes.
    It always creates ``how_many_individuals`` children.

    :param how_many_individuals: The number of individuals to create in the offspring (can't exceed population size passed to cross method and can't be less than 2).
    :param probability: The probability of performing the crossover operation.
    :param alfa: The parameter used for blending two genes, must be positive, preferably closer to 0 and less than 1.
    :raises ValueError: If the alfa parameter is not positive or how_many_individuals is less than 2.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(
            self,
            how_many_individuals: int,  # count of the created offspring and the number of parents to use in the crossover, cant be more than the population size
            probability: float = 0,
            alfa: float = 0.2  # parameter passed down internally to the BlendCrossoverAlfa, used by this crossover
    ):
        if how_many_individuals < 2:
            raise ValueError("The how_many_individuals parameter must be greater than 2 or equal (and not larger than the population size passed to the corss method).")

        super().__init__(how_many_individuals, probability)
        self.alfa = alfa

    def _cross(self, population_parent: Population) -> Population:
        """
        :return: The offspring population of size ``how_many_individuals`` defined in the constructor of the class.
        """
        parent_indexes = []
        for i in range(self.how_many_individuals):
            index = np.random.randint(population_parent.population_size)
            while index in parent_indexes:
                index = np.random.randint(population_parent.population_size)
            parent_indexes.append(index)

        # assume all individuals have the same number of genes in their chromosome
        len_of_chromosome = len(population_parent.population[0].chromosome)

        blend_crossover = BlendCrossoverAlfa(how_many_individuals=1,  # unused
                                             probability=1,  # unused
                                             alfa=self.alfa
                                             )

        children = Population(population=[])
        for j in range(self.how_many_individuals):
            child = Individual(chromosome=np.empty(len_of_chromosome, dtype=float), value=0)

            for i in range(len_of_chromosome):
                k = np.random.randint(0, self.how_many_individuals - 1)
                while k == j:
                    k = np.random.randint(0, self.how_many_individuals - 1)

                child.chromosome[i] = blend_crossover.blend_one_pair_of_genes(
                    population_parent.population[parent_indexes[j]].chromosome[i],
                    population_parent.population[parent_indexes[k]].chromosome[i]
                )

            children.add_to_population(child)
        return children
