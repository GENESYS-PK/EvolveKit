from dataclasses import dataclass

from evolvekit.core.Ga.GaIndividual import GaIndividual


@dataclass
class GaStatistics:
    generation: int = 1
    stagnation: int = 0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0
    best_indiv: GaIndividual | None = None
    worst_indiv: GaIndividual | None = None
    start_time: float = 0.0
    last_time: float = 0.0
