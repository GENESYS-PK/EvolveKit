import numpy as np
from core import Population
from core.operators import Selection

class RankSelection(Selection):
    """
    Rank selection. Internally, it uses stochastic acceptance roulette-wheel algorithm.
    For more details about stochastic acceptance roulette-wheel see: https://arxiv.org/abs/1109.3627

    :param target_population: The number of individuals to select.
    :param maximize: Set "true" if higher scores from fitness function are better.
    """

    allowed_representation = [
        "real",
        "binary"
    ]

    def __init__(self, target_population: int, maximize: bool):
        super().__init__(target_population, maximize)

    def _select(self, population: Population) -> Population:
        selected_individuals = []
        pop_size = population.population_size

        sorted_pop = sorted(population.population, key = lambda obj: obj.value, reverse = not self.maximize)
        for i in range(pop_size):
            sorted_pop[i].value = i + 1

        while len(selected_individuals) < self.target_population:
            indiv = population.population[np.random.randint(0, pop_size)]
            weight = indiv.value / pop_size

            if np.random.rand() < weight:
                selected_individuals.append(indiv)

        return Population(population=selected_individuals)
