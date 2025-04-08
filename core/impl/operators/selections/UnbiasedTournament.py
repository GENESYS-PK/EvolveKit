import numpy as np
from core import Population
from core.operators import Selection


class UnbiasedTournament(Selection):
    """
    Implements unbiased tournament selection.
    For more details see: https://www.cs.colostate.edu/~genitor/2005/GECCO247.pdf

    :param target_population: The number of individuals to select. Must be set to be exactly the number of individuals in the whole population.
    :param maximize: Set "true" if higher scores from fitness function are better.
    """

    allowed_representation = [
        "real",
        "binary",
    ]

    def __init__(
        self, target_population: int, maximize: bool
    ):
        super().__init__(target_population, maximize)

    def _select(self, population: Population) -> Population:
        if len(population.population) != self.target_population:
            raise RuntimeError("In UnbiasedTournament, target_population must be set to be exactly the number of individuals in the whole population")

        selected_individuals = []
        shuffled_pop = np.random.permutation(population.population)

        if self.maximize:
            for a, b in zip(population.population, shuffled_pop):
                selected_individuals.append(a if a.value > b.value else b)
        else:
            for a, b in zip(population.population, shuffled_pop):
                selected_individuals.append(a if a.value < b.value else b)

        return Population(population=selected_individuals)
