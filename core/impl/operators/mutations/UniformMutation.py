import numpy as np
from core.Population import Population
from core.operators.Mutation import Mutation
from core.Individual import Individual
from core.Representation import Representation


class UniformMutation(Mutation):
    """
    The class implements Uniform mutation, where a random gene in the chromosome is selected,
    and its value is replaced with another random value from the domain.

     Algorytmy Genetyczne kompedium tom II Operator mutacji dla problemów numerycznych. Tomasz Gwiazda pp. 182-183.

    :param probability: The probability of performing the mutation operation.
    :param domain: The domain of gene
    :param allowed_representation: A list of allowed representations for the mutation operation.
    """

    allowed_representation = [Representation.REAL]

    def __init__(self, domain: np.ndarray[float], probability: float = 0):
        super().__init__(probability)
        self.domain = domain

    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        :param individual: The individual to perform the mutation operation on.
        :param population: The population to which the individual belongs.
        :returns: None
        """
        mutation_point = np.random.randint(0, len(individual.chromosome) - 1)
        new_value_of_gene = np.random.uniform(self.domain[0], self.domain[1])
        individual.chromosome[mutation_point] = new_value_of_gene
