from dataclasses import dataclass

from evolvekit.core.Ga.GaIndividual import GaIndividual


@dataclass
class GaStatistics:
    generation: int
    stagnation: int
    mean: float
    median: float
    stdev: float
    best_indiv: GaIndividual | None
    worst_indiv: GaIndividual | None
    start_time: float
    last_time: float
