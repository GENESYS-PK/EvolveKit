import numpy as np
from core import Population
from core.operators import Selection

class SaRouletteSigma(Selection):
    """
    Stochastic acceptance roulette-wheel selection with sigma scaling.
    For more details about stochastic acceptance roulette-wheel see: https://arxiv.org/abs/1109.3627
    For more details about sigma scaling see: https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Practical%20Handbook%20of%20GENETIC%20ALGORITHMS%2C%20Volume%20II/ganf3.pdf

    :param target_population: The number of individuals to select.
    :param maximize: Set "true" if higher scores from fitness function are better.
    :param offset: Adds constant value to each individual's score. Offset should be non-negative.
    :param minimum: Minimum score after scaling. Should be non-negative.
    :param factor: Constant factor in the denominator. Should be greater than zero.
    :param epsilon: Standard deviation threshold below which all individuals are treated as if they have equal scores. This should be small value, but greater than zero.
    """

    allowed_representation = [
        "real",
        "binary"
    ]

    def __init__(self, target_population: int, maximize: bool, offset: float = 1.0, minimum: float = 0.1, factor: float = 2.0, epsilon: float = 0.0001):
        super().__init__(target_population, maximize)
        self.offset = offset
        self.minimum = minimum
        self.factor = factor
        self.epsilon = epsilon

    def _select(self, population: Population) -> Population:
        selected_individuals = []
        pop_size = population.population_size
        scores = np.array([obj.value for obj in population.population])
        stdev = np.std(scores)
        mean = np.mean(scores)

        if stdev < self.epsilon:
            while len(selected_individuals) < self.target_population:
                indiv = population.population[np.random.randint(0, pop_size)]
                selected_individuals.append(indiv)
            return Population(population=selected_individuals)

        c = 1.0 / (stdev * self.factor)
        maxDiff = max(scores) - mean if self.maximize else mean - min(scores)
        scale = maxDiff * c + self.offset

        while len(selected_individuals) < self.target_population:
            indiv = population.population[np.random.randint(0, pop_size)]
            diff = indiv.value - mean if self.maximize else mean - indiv.value
            weight = max(diff * c + self.offset, self.minimum)

            if np.random.rand() * scale < weight:
                selected_individuals.append(indiv)

        return Population(population=selected_individuals)
