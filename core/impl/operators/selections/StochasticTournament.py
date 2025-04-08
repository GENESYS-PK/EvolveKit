import numpy as np
from core import Population
from core.operators import Selection


class StochasticTournament(Selection):
    """
    Implements stochastic tournament selection.

    :param target_population: The number of individuals to select.
    :param maximize: Set "true" if higher scores from fitness function are better.
    :param tournament_size: The number of individuals that compete in each tournament.
    :param p: Probability to select the best individual in the tournament. Must be in the range [0, 1].
    """

    allowed_representation = [
        "real",
        "binary",
    ]

    def __init__(
        self, target_population: int, maximize: bool, tournament_size: int = 3, p: float = 0.8
    ):
        super().__init__(target_population, maximize)
        self.tournament_size = tournament_size
        self.p = p

    def _select(self, population: Population) -> Population:
        selected_individuals = []
        while len(selected_individuals) < self.target_population:
            indices = np.random.choice(len(population.population), size=self.tournament_size, replace=False)
            indiv = np.array(population.population)[indices]
            indiv = sorted(indiv, key=lambda obj: obj.value, reverse=self.maximize)
            threshold = self.p

            for crr in indiv:
                if np.random.random() < threshold:
                    selected_individuals.append(crr)
                threshold = threshold * (1.0 - self.p)

        return Population(population=selected_individuals)
