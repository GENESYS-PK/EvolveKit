import numpy as np
from core import Population
from core.operators import Selection


class TournamentSelection(Selection):
    """
    Implements tournament selection.

    :param target_population: The number of individuals to select.
    :param maximize: Set "true" if higher scores from fitness function are better.
    :param tournament_size: The number of individuals that compete in each tournament.
    """

    allowed_representation = [
        "real",
        "binary",
    ]

    def __init__(
        self, target_population: int, maximize: bool, tournament_size: int = 3
    ):
        super().__init__(target_population, maximize)
        self.tournament_size = tournament_size

    def _select(self, population: Population) -> Population:
        selected_individuals = []
        pop_size = population.population_size

        while len(selected_individuals) < self.target_population:
            bestValue = float('-inf') if self.maximize else float('inf')
            bestIdx = 0

            for _ in range(self.tournament_size):
                idx = np.random.randint(0, pop_size)
                value = population.population[idx].value

                if self.maximize:
                    if value > bestValue:
                        bestValue = value
                        bestIdx = idx
                else:
                    if value < bestValue:
                        bestValue = value
                        bestIdx = idx

            selected_individuals.append(population.population[bestIdx])

        return Population(population=selected_individuals)
