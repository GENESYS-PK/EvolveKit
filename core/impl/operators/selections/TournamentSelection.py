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

        while len(selected_individuals) < self.target_population:
            # Losowy wybór osobników do turnieju
            pop_size = population.population_size
            index_1 = np.random.randint(0, pop_size)
            index_2 = np.random.randint(0, pop_size)
            tournament = [
                population.population[index_1],
                population.population[index_2],
            ]

            # Wyłonienie najlepszego osobnika w turnieju
            if self.maximize:
                best_individual = max(tournament, key=lambda ind: ind.value)
            else:
                best_individual = min(tournament, key=lambda ind: ind.value)

            selected_individuals.append(best_individual)

        # Utworzenie nowej populacji z wybranych osobników
        return Population(population=selected_individuals)
