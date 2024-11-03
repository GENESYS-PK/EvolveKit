from dataclasses import dataclass
from core.Population import Population


@dataclass
class EvolutionState:
    """
    A data class that holds the state of the evolution process. This class stores the
    current population, selected population, new population, and tracks relevant
    information such as whether the evolution is maximizing or minimizing and the
    current epoch.
    """

    current_population: Population
    selected_population: Population | None
    new_population: Population | None
    maximize: bool
    population_size: int
    current_epoch: int = 0

    def update_evolution_state(self) -> None:
        """
        Update the state of the evolution after a complete cycle. This method moves the state to the
        next epoch by updating the current population, clearing the selected and new populations,
        and incrementing the epoch count.
        """
        self.current_population = (
            self.new_population if self.new_population else self.current_population
        )
        self.selected_population = None
        self.new_population = None
        self.current_epoch += 1
