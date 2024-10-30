from core.ClampStrategy import ClampStrategy
from typing import Sequence
from core.Population import Population
import numpy as np


class SimpleClampStrategy(ClampStrategy):
    """
    A simple clamp strategy that constrains individuals' values
    to be within the specified domains.
    """

    def clamp(self, variable_domains: Sequence[tuple[float, float]], population: Population) -> None:
        for individual in population.individuals:
            for i, (low, high) in enumerate(variable_domains):
                # Clamp the values to the specified bounds
                individual.chromosome[i] = np.clip(individual.chromosome[i], low, high)