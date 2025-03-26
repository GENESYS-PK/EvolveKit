import numpy as np
from core import Population
from core.operators import Selection

class StochasticAcceptanceRoulette(Selection):
    """
    Stochastic acceptance roulette-wheel selection with windowing.
    For more details about stochastic acceptance roulette-wheel see: https://arxiv.org/abs/1109.3627
    For more details about windowing see: https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Practical%20Handbook%20of%20GENETIC%20ALGORITHMS%2C%20Volume%20II/ganf3.pdf

    :param target_population: The number of individuals to select.
    :param maximize: Set "true" if higher scores from fitness function are better.
    :param offset: Adds constant value to each individual's score. Higher offset means lower selection pressure. Offset should be non-negative.
    """

    allowed_representation = [
        "real",
        "binary"
    ]

    def __init__(self, target_population: int, maximize: bool, offset: float = 0.25):
        super().__init__(target_population, maximize)
        self.offset = offset

    def _select(self, population: Population) -> Population:
        selected_individuals = []
        pop_size = population.population_size
        max_val = max(population.population, key = lambda obj: obj.value).value
        min_val = min(population.population, key = lambda obj: obj.value).value
        div = max_val - min_val + self.offset
        sub = min_val - self.offset if self.maximize else max_val + self.offset

        while len(selected_individuals) < self.target_population:
            indiv = population.population[np.random.randint(0, pop_size)]
            diff = indiv.value - sub if self.maximize else sub - indiv.value
            weight = diff / div

            if np.random.rand() < weight:
                selected_individuals.append(indiv)

        return Population(population=selected_individuals)
