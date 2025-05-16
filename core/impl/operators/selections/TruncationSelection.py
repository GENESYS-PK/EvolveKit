import numpy as np
from core import Population
from core.operators import Selection

class TruncationSelection(Selection):
    """
    Implements truncation selection.

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

        if pop_size <= self.target_population:
            raise RuntimeError("Target population must be smaller than the whole population")

        sorted_pop = sorted(population.population, key = lambda obj: obj.value)

        if self.maximize:
            for indiv in reversed(sorted_pop):
                if len(selected_individuals) >= self.target_population:
                    break
                selected_individuals.append(indiv)
        else:
            for indiv in sorted_pop:
                if len(selected_individuals) >= self.target_population:
                    break
                selected_individuals.append(indiv)

        return Population(population=selected_individuals)
