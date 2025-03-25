import numpy as np
from core import Population
from core.operators import Selection


class TournamentSelection(Selection):
    """
    Selekcja turniejowa, która wybiera osobniki poprzez losowe grupowanie ich w turnieje i wybieranie najlepszego
    osobnika w każdym turnieju.

    :param tournament_size: Rozmiar turnieju (liczba osobników w jednym turnieju).
    """

    allowed_representation = [
        "real",
        "binary",
    ]  # Przykładowo może działać z różnymi reprezentacjami

    def __init__(
        self, target_population: int, maximize: bool, tournament_size: int = 3
    ):
        super().__init__(target_population, maximize)
        self.tournament_size = tournament_size

    def _select(self, population: Population) -> Population:
        """
        Przeprowadza selekcję turniejową na populacji.

        :param population: Populacja, na której wykonywana jest selekcja.
        :returns: Nowa populacja po selekcji.
        """
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

        # Utworzenie nowej populacji z wybranych osobników
        return Population(population=selected_individuals)
